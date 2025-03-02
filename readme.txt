CLIMB: Controllable Longitudinal Brain Image Generation via Mamba-based Latent Diffusion Model

📦 Installation

Clone the repository and install dependencies:

git clone https://github.com/duyphuongcri/CLIMB.git
cd CLIMB
pip install -r requirements.txt

🔥 Usage

-----Train models------
Step1: Train autoencoder model
python train_autoencoder.py 

Step2: Extract latent features
python extract_latents.py

Step3: Train diffusion model conditioned on variables (age, gender, disease, status, biomarker,...)
python train_diffusion_variables.py

Step4: Train diffusion model with all conditional factors (variables and image features)
python train_diffusion_image_features.py

Step5: Train IRLSTM model (for predicting brain volumes structure and disease status at the projected age)
python train_irlstm.py 

-----Evaluating Model------
python measure_performance.py 

---- Inference ------
python inference.py 

📂 Repository Structure

CLIMB/
│── configs/         # Configuration files for different experiments
│── datasets/        # Data preprocessing and loading scripts
│── models/          # Implementation of various CIL models
│── utils/           # Helper functions and utilities
│── main.py          # Entry point for training and evaluation
│── README.md        # This file
│── requirements.txt # List of dependencies

📊 Datasets

ADNI 

📜 Citation

If you use CLIMB in your research, please cite:

@article{CLIMB,
  author    = {Author Name},
  title     = {CLIMB: Controllable Longitudinal Brain Image Generation via Mamba-based Latent Diffusion Model},
  year      = {2025}
}

🤝 Contributing

We welcome contributions! Feel free to open an issue or a pull request.

📬 Contact

For any questions or support, open an issue or contact at duyphuongcri@gmail.com

