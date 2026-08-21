#!/bin/bash
# Script thiet lap systemd service mlops-serve tren GCE VM
BUCKET_NAME=${1:-"mlops-wine-vda-0231929600"}

sudo bash -c "cat <<EOF > /etc/systemd/system/mlops-serve.service
[Unit]
Description=MLOps Model Inference Server
After=network.target

[Service]
User=$USER
WorkingDirectory=/home/$USER
Environment=\"GCS_BUCKET=$BUCKET_NAME\"
Environment=\"GOOGLE_APPLICATION_CREDENTIALS=/home/$USER/sa-key.json\"
ExecStart=/usr/bin/python3 /home/$USER/src/serve.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF"

sudo systemctl daemon-reload
sudo systemctl enable mlops-serve
sudo systemctl restart mlops-serve

echo ">>> Systemd service mlops-serve da duoc cau hinh thanh cong voi Bucket: $BUCKET_NAME!"
