mkdir -p ~/.streamlit/
mkdir -p ~/nltk_data

echo "\
[general]
email = \"ezra.vergabera@gmail.com\"
" > ~/.streamlit/credentials.toml

echo "\
[server]
headless = true
enableCORS = false
port = $PORT
" > ~/.streamlit/config.toml