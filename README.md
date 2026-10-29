# DDoS Attack Detection Using Big Data

Streamlit + pandas (chunk processing) + scikit-learn (Random Forest).

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Host on Render (free)
1. Push these files to a GitHub repository (app.py, requirements.txt, render.yaml).
2. On render.com: New + -> Blueprint (or Web Service) -> connect the repo.
3. If you choose Web Service manually:
   - Runtime: Python 3
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false`
   - Environment variable: PYTHON_VERSION = 3.11.9
4. Click Deploy. Your public link will look like https://ddos-detection.onrender.com

Edit STUDENT_NAME, ROLL_NO and CLASS_NAME at the top of app.py.
