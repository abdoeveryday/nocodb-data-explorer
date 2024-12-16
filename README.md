# NoCodeB Data Explorer

A Flask-based web application for exploring and analyzing product data from NoCodeB. The application implements lazy loading for efficient data handling and provides visualization tools for better data understanding.

## Features

- Lazy loading of data with caching mechanism
- Interactive data table with pagination
- Data visualizations including:
  - Stock level distribution
  - Product family distribution
  - Sales vs Stock correlation
- AI-powered chat interface for data queries using OpenAI GPT
- Error handling and retry mechanisms for API requests

## Setup

1. Clone the repository:
```bash
git clone [your-repository-url]
cd nocodb-import
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key
NOCODB_API_TOKEN=your_nocodb_api_token
```

4. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `NOCODB_API_TOKEN`: Your NoCodeB API token

## Project Structure

- `app.py`: Main application file
- `templates/`: HTML templates
- `requirements.txt`: Python dependencies
- `.env`: Environment variables (not tracked in git)
- `.gitignore`: Git ignore rules

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
