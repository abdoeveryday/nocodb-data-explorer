import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, jsonify, render_template, request
import requests
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import openai
import logging
import time
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set OpenAI API key and NoCodeDB API token from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')
NOCODB_API_TOKEN = os.getenv('NOCODB_API_TOKEN')
OPENAI_EMBEDDINGS_API_KEY = os.getenv('OPENAI_EMBEDDINGS_API_KEY')

if not openai.api_key:
    raise ValueError("OpenAI API key not found in environment variables")
if not NOCODB_API_TOKEN:
    raise ValueError("NoCodeB API token not found in environment variables")
if not OPENAI_EMBEDDINGS_API_KEY:
    raise ValueError("OpenAI Embeddings API key not found in environment variables")

app = Flask(__name__)

# Global variables
ITEMS_PER_PAGE = 100  # Increased for better performance
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

def fetch_page(page, retries=0):
    """Fetch a single page of data with retry logic."""
    try:
        time.sleep(0.5)  # Add a 500ms delay between requests
        response = requests.get(
            'https://app.nocodb.com/api/v2/tables/mgy5ln86ajw5sm5/records',
            params={'page': page, 'limit': ITEMS_PER_PAGE},
            headers={'xc-token': NOCODB_API_TOKEN}
        )
        
        if response.status_code == 429:  # Rate limit hit
            logger.warning("Rate limit hit, waiting for 5 seconds...")
            time.sleep(5)
            return fetch_page(page, retries)
            
        if response.status_code == 502 and retries < MAX_RETRIES:
            logger.warning(f"502 error encountered for page {page}, attempt {retries + 1}/{MAX_RETRIES}")
            time.sleep(RETRY_DELAY)
            return fetch_page(page, retries + 1)
            
        response.raise_for_status()
        return response.json()
    except Exception as e:
        if retries < MAX_RETRIES:
            logger.warning(f"Error fetching page {page}, attempt {retries + 1}/{MAX_RETRIES}: {str(e)}")
            time.sleep(RETRY_DELAY)
            return fetch_page(page, retries + 1)
        raise

@lru_cache(maxsize=100)
def get_page_data(page):
    """Get data for a specific page with caching."""
    try:
        data = fetch_page(page)
        if 'list' not in data:
            logger.error(f"Unexpected API response format. Available keys: {data.keys()}")
            raise KeyError("API response does not contain 'list' key")
        return data['list'], data.get('pageInfo', {}).get('totalRows', 0)
    except Exception as e:
        logger.error(f"Error fetching page {page}: {str(e)}")
        return [], 0

def get_total_records():
    """Get the total number of records."""
    _, total = get_page_data(1)
    return total

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_table_data')
def get_table_data():
    try:
        page = int(request.args.get('page', 1))
        records, total_records = get_page_data(page)
        total_pages = (total_records + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
        
        return jsonify({
            'data': records,
            'current_page': page,
            'total_pages': total_pages,
            'total_records': total_records
        })
    except Exception as e:
        logger.error(f"Error in get_table_data: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/get_data')
def get_data():
    try:
        # Get first page for basic statistics
        records, _ = get_page_data(1)
        df = pd.DataFrame(records)
        
        # Stock Level Distribution
        stock_fig = px.histogram(df, x='Qte_stock', title='Stock Level Distribution (Sample)',
                            labels={'Qte_stock': 'Stock Quantity'})
        stock_fig.update_layout(showlegend=False)
        
        # Product Family Distribution
        family_dist = df['famille'].value_counts()
        family_fig = px.pie(values=family_dist.values, names=family_dist.index,
                        title='Product Family Distribution (Sample)')
        
        # Summary Statistics Table
        summary_stats = pd.DataFrame({
            'Metric': ['Products in View', 'Average Stock', 'Average Sales'],
            'Value': [len(df), df['Qte_stock'].mean(), df['Qte_ventes'].mean()]
        })
        summary_fig = go.Figure(data=[go.Table(
            header=dict(values=list(summary_stats.columns)),
            cells=dict(values=[summary_stats[col] for col in summary_stats.columns])
        )])
        
        # Sales vs Stock Correlation
        scatter_fig = px.scatter(df, x='Qte_stock', y='Qte_ventes',
                            title='Sales vs Stock Correlation (Sample)',
                            labels={'Qte_stock': 'Stock Quantity',
                                    'Qte_ventes': 'Sales Quantity'})
        
        return jsonify({
            'stock_chart': {'data': stock_fig.data, 'layout': stock_fig.layout},
            'family_dist': {'data': family_fig.data, 'layout': family_fig.layout},
            'summary_table': {'data': summary_fig.data, 'layout': summary_fig.layout},
            'scatter_plot': {'data': scatter_fig.data, 'layout': scatter_fig.layout}
        })
    except Exception as e:
        logger.error(f"Error in get_data: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        query = request.json.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400
            
        # Get first few pages for the chat context
        all_records = []
        for page in range(1, 4):  # Get first 3 pages
            records, _ = get_page_data(page)
            all_records.extend(records)
            
        df = pd.DataFrame(all_records)
        
        # Create embeddings for the sample data
        texts = df.apply(lambda row: f"SKU: {row['SKU']}, Designation: {row['designation']}, "
                                    f"Family: {row['famille']}, Stock Quantity: {row['Qte_stock']}, "
                                    f"Sales Quantity: {row['Qte_ventes']}, Color: {row['couleur']}", axis=1).tolist()
        
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_EMBEDDINGS_API_KEY)
        vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
        
        # Initialize QA chain
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai.api_key)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        
        # Get response from QA chain
        result = qa_chain({"question": query, "chat_history": []})
        response = result['answer']
        
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        logger.info("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}", exc_info=True)
