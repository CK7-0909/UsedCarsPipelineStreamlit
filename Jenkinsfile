pipeline {
    agent any
    stages {
        stage('Setup Python Environment') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install -r requirements.txt
                '''
            }
        }
        stage('Deploy Streamlit') {
            steps {
                sh '''
                # Kill old Streamlit process if running
                pkill -f "streamlit run" || true

                # Wait briefly
                sleep 3

                # Start Streamlit in background using nohup so it stays alive after Jenkins finishes
                nohup venv/bin/python3 -m streamlit run main.py \
                    --server.port 8501 --server.address 0.0.0.0 \
                    > streamlit.log 2>&1 &
                '''
            }
        }
    }
}
