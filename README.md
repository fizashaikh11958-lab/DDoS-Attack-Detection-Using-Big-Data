# DDoS Attack Detection Using Big Data (Streamlit)

Reads a big traffic file in chunks, detects SYN / UDP / HTTP / ICMP floods with a
Random Forest model and raises a DDoS alert.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Host on Streamlit Community Cloud (free)
1. Upload this folder's files to a GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Create app -> choose repo and branch -> Main file path: app.py -> Deploy.

Edit STUDENT_NAME, ROLL_NO and CLASS_NAME at the top of app.py.
