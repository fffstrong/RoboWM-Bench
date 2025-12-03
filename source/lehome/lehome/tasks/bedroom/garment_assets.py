"""Garment asset paths configuration for different garment types."""

import os

# Base path for garment assets
GARMENT_BASE_PATH = os.path.join(os.getcwd(), "Assets/objects/garment")

# Garment asset paths dictionary
# Each garment type has 5 assets (indices 0-4)
GARMENT_ASSETS = {
    "top-long-sleeve": [
        os.path.join(GARMENT_BASE_PATH, "Tops/Collar_Lsleeve_FrontClose/TCLC_002/TCLC_002_obj.usd"),
        os.path.join(GARMENT_BASE_PATH, "Tops/Collar_Lsleeve_FrontClose/TCLC_007/TCLC_007_obj.usd"),
        os.path.join(GARMENT_BASE_PATH, "Tops/Collar_Lsleeve_FrontClose/TCLC_074/TCLC_074_obj.usd"),
        os.path.join(GARMENT_BASE_PATH, "Tops/Collar_Lsleeve_FrontClose/TCLC_082/TCLC_082_obj.usd"),
        os.path.join(GARMENT_BASE_PATH, "Tops/Collar_Lsleeve_FrontClose/TCLC_Shirt028/TCLC_Shirt028_obj.usd"),
    ],
    "top-short-sleeve": [
        os.path.join(GARMENT_BASE_PATH, "Tops/Collar_Ssleeve_FrontClose/TCSC_075/TCSC_075_obj.usd"),
        os.path.join(GARMENT_BASE_PATH, "Tops/Collar_Ssleeve_FrontClose/TCSC_model2_032/TCSC_model2_032_obj.usd"),
        os.path.join(GARMENT_BASE_PATH, "Tops/Collar_Ssleeve_FrontClose/TCSC_T-Shirt003/TCSC_T-Shirt003_obj.usd"),
        os.path.join(GARMENT_BASE_PATH, "Tops/Collar_Ssleeve_FrontClose/TCSC_polo004/TCSC_polo004_obj.usd"),
        os.path.join(GARMENT_BASE_PATH, "Tops/Collar_Ssleeve_FrontClose/TCSC_083/TCSC_083_obj.usd"),
    ],
    "short-pant": [],  # Reserved for future implementation
    "long-pant": [],   # Reserved for future implementation
}

# Valid garment types
VALID_GARMENT_TYPES = list(GARMENT_ASSETS.keys())

