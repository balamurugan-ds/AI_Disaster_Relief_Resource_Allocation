import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

from src.allocation import (
    calculate_priority,
    priority_level,
    allocate_resource
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Disaster Relief Assistant",
    page_icon="🚨",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: bold;
    text-align: center;
}
.subtitle {
    text-align: center;
    font-size: 17px;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD ML MODELS
# IMPORTANT: These filenames must match your saved model files.
# ============================================================

@st.cache_resource
def load_ml_models():

    models = {}

    models["food"] = joblib.load(
        "models/food_required_model.pkl"
    )

    models["water"] = joblib.load(
        "models/water_required_model.pkl"
    )

    models["medicine"] = joblib.load(
        "models/medical_supplies_required_model.pkl"
    )

    models["rescue"] = joblib.load(
        "models/rescue_teams_required_model.pkl"
    )

    models["shelter"] = joblib.load(
        "models/shelter_required_model.pkl"
    )

    return models


# ============================================================
# LOAD DEEP LEARNING MODEL
# ============================================================

@st.cache_resource
def load_dl_model():

    return tf.keras.models.load_model(
        "models/disaster_relief_ann.keras"
    )


# ============================================================
# LOAD DL PREPROCESSOR
# ============================================================

@st.cache_resource
def load_preprocessor():

    return joblib.load(
        "models/dl_preprocessor.pkl"
    )


# ============================================================
# LOAD ALL MODELS
# ============================================================

try:

    ml_models = load_ml_models()
    dl_model = load_dl_model()
    dl_preprocessor = load_preprocessor()

except Exception as e:

    st.error(
        "Model files could not be loaded. "
        "Check your models folder and run the training scripts."
    )

    st.exception(e)
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🚨 AI Disaster Relief Resource Allocation'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered disaster analysis, resource prediction '
    'and resource allocation system'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR - DISASTER INFORMATION
# ============================================================

st.sidebar.title("⚙️ Disaster Information")

disaster_type = st.sidebar.selectbox(
    "Disaster Type",
    [
        "Flood",
        "Earthquake",
        "Cyclone",
        "Landslide",
        "Drought",
        "Wildfire"
    ]
)

location = st.sidebar.selectbox(
    "Location",
    [
        "Chennai",
        "Coimbatore",
        "Tiruppur",
        "Madurai",
        "Trichy",
        "Salem",
        "Erode",
        "Thanjavur",
        "Tirunelveli",
        "Cuddalore"
    ]
)

population_affected = st.sidebar.number_input(
    "Population Affected",
    min_value=100,
    max_value=1000000,
    value=10000,
    step=500
)

disaster_severity = st.sidebar.slider(
    "Disaster Severity",
    min_value=1,
    max_value=10,
    value=5
)

injuried_people = st.sidebar.number_input(
    "Injuried People",
    min_value=0,
    max_value=100000,
    value=200,
    step=10
)

displaced_people = st.sidebar.number_input(
    "Displaced People",
    min_value=0,
    max_value=500000,
    value=5000,
    step=100
)

duration_days = st.sidebar.number_input(
    "Expected Duration (Days)",
    min_value=1,
    max_value=365,
    value=5
)

temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=-20.0,
    max_value=60.0,
    value=30.0
)

rainfall = st.sidebar.number_input(
    "Rainfall",
    min_value=0.0,
    max_value=2000.0,
    value=100.0
)

infrastructure_damage = st.sidebar.slider(
    "Infrastructure Damage (%)",
    min_value=0.0,
    max_value=100.0,
    value=30.0
)


# ============================================================
# RESOURCE INVENTORY
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("📦 Available Resources")

available_food = st.sidebar.number_input(
    "Available Food Packets",
    min_value=0,
    value=15000
)

available_water = st.sidebar.number_input(
    "Available Water (Litres)",
    min_value=0,
    value=30000
)

available_medicine = st.sidebar.number_input(
    "Available Medical Supplies",
    min_value=0,
    value=3000
)

available_rescue = st.sidebar.number_input(
    "Available Rescue Teams",
    min_value=0,
    value=10
)

available_shelter = st.sidebar.number_input(
    "Available Shelter Capacity",
    min_value=0,
    value=5000
)


# ============================================================
# INPUT DATAFRAME
# EXACT FEATURE NAMES FROM YOUR DATASET
# ============================================================

input_data = pd.DataFrame({

    "disaster_type": [disaster_type],

    "location": [location],

    "population_affected": [
        population_affected
    ],

    "disaster_severity": [
        disaster_severity
    ],

    "injuried_people": [
        injuried_people
    ],

    "displaced_people": [
        displaced_people
    ],

    "duration_days": [
        duration_days
    ],

    "temperature": [
        temperature
    ],

    "rainfall": [
        rainfall
    ],

    "infrastructure_damage": [
        infrastructure_damage
    ]

})


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔮 Predict Disaster Requirements",
    use_container_width=True
)


if predict_button:

    # ========================================================
    # MACHINE LEARNING PREDICTIONS
    # ========================================================

    try:

        food_ml = float(
            np.asarray(
                ml_models["food"].predict(input_data)
            ).reshape(-1)[0]
        )

        water_ml = float(
            np.asarray(
                ml_models["water"].predict(input_data)
            ).reshape(-1)[0]
        )

        medicine_ml = float(
            np.asarray(
                ml_models["medicine"].predict(input_data)
            ).reshape(-1)[0]
        )

        rescue_ml = float(
            np.asarray(
                ml_models["rescue"].predict(input_data)
            ).reshape(-1)[0]
        )

        shelter_ml = float(
            np.asarray(
                ml_models["shelter"].predict(input_data)
            ).reshape(-1)[0]
        )

        # ================================================
        # DEEP LEARNING PREDICTIONS
        # ================================================

        processed_input = dl_preprocessor.transform(
            input_data
        )

        dl_prediction = np.asarray(
            dl_model.predict(
                processed_input,
                verbose=0
            )
        ).reshape(-1)

        if len(dl_prediction) != 5:
            st.error(
                "ANN model must output five predictions "
                "in this order: food, water, medical supplies, "
                "rescue teams and shelter."
            )
            st.stop()

        food_dl = max(0.0, float(dl_prediction[0]))
        water_dl = max(0.0, float(dl_prediction[1]))
        medicine_dl = max(0.0, float(dl_prediction[2]))
        rescue_dl = max(1.0, float(dl_prediction[3]))
        shelter_dl = max(0.0, float(dl_prediction[4]))

    except Exception as e:

        st.error(
            "Prediction failed. Your saved models and "
            "preprocessor must use the same feature names "
            "and feature order as the training dataset."
        )

        st.exception(e)
        st.stop()


    # ========================================================
    # FINAL PREDICTIONS - ML + DL AVERAGE
    # ========================================================

    food_required = max(
        0,
        int(round((food_ml + food_dl) / 2))
    )

    water_required = max(
        0,
        int(round((water_ml + water_dl) / 2))
    )

    medical_supplies_required = max(
        0,
        int(round((medicine_ml + medicine_dl) / 2))
    )

    rescue_teams_required = max(
        1,
        int(round((rescue_ml + rescue_dl) / 2))
    )

    shelter_required = max(
        0,
        int(round((shelter_ml + shelter_dl) / 2))
    )


    # ========================================================
    # PRIORITY SCORE
    # ========================================================

    priority_score = calculate_priority(

        severity=disaster_severity,

        population_affected=population_affected,

        injuried_people=injuried_people,

        displaced_people=displaced_people,

        infrastructure_damage=infrastructure_damage

    )

    priority = priority_level(
        priority_score
    )


    # ========================================================
    # RESOURCE ALLOCATION
    # ========================================================

    allocated_food = allocate_resource(
        food_required,
        available_food,
        priority_score
    )

    allocated_water = allocate_resource(
        water_required,
        available_water,
        priority_score
    )

    allocated_medicine = allocate_resource(
        medical_supplies_required,
        available_medicine,
        priority_score
    )

    allocated_rescue = allocate_resource(
        rescue_teams_required,
        available_rescue,
        priority_score
    )

    allocated_shelter = allocate_resource(
        shelter_required,
        available_shelter,
        priority_score
    )


    # ========================================================
    # DISASTER ASSESSMENT
    # ========================================================

    st.markdown("---")
    st.header("🚨 Disaster Assessment")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Priority Score",
        f"{priority_score}/100"
    )

    col2.metric(
        "Priority Level",
        str(priority)
    )

    col3.metric(
        "Affected Population",
        f"{population_affected:,}"
    )


    # ========================================================
    # PREDICTED RESOURCE REQUIREMENTS
    # ========================================================

    st.header("📊 Predicted Resource Requirements")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "🍚 Food Packets",
        f"{food_required:,}"
    )

    col2.metric(
        "💧 Water (Litres)",
        f"{water_required:,}"
    )

    col3.metric(
        "💊 Medical Supplies",
        f"{medical_supplies_required:,}"
    )

    col4.metric(
        "🚑 Rescue Teams",
        f"{rescue_teams_required:,}"
    )

    col5.metric(
        "🏠 Shelter Capacity",
        f"{shelter_required:,}"
    )


    # ========================================================
    # RESOURCE REQUIREMENT ANALYSIS - MATPLOTLIB
    # ========================================================

    resource_df = pd.DataFrame({

        "Resource": [
            "Food",
            "Water",
            "Medical Supplies",
            "Rescue Teams",
            "Shelter"
        ],

        "Required": [
            food_required,
            water_required,
            medical_supplies_required,
            rescue_teams_required,
            shelter_required
        ]

    })

    st.subheader("📈 Resource Requirement Analysis")

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        resource_df["Resource"],
        resource_df["Required"]
    )

    ax.set_title("Predicted Resource Requirements")
    ax.set_xlabel("Resource")
    ax.set_ylabel("Required Quantity")

    ax.bar_label(
        bars,
        fmt="%.0f",
        padding=3
    )

    ax.tick_params(
        axis="x",
        labelrotation=15
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


    # ========================================================
    # RESOURCE ALLOCATION TABLE
    # ========================================================

    st.header("📦 Resource Allocation")

    allocation_df = pd.DataFrame({

        "Resource": [
            "Food Packets",
            "Water (Litres)",
            "Medical Supplies",
            "Rescue Teams",
            "Shelter Capacity"
        ],

        "Required": [
            food_required,
            water_required,
            medical_supplies_required,
            rescue_teams_required,
            shelter_required
        ],

        "Available": [
            available_food,
            available_water,
            available_medicine,
            available_rescue,
            available_shelter
        ],

        "Allocated": [
            allocated_food,
            allocated_water,
            allocated_medicine,
            allocated_rescue,
            allocated_shelter
        ]

    })

    st.dataframe(
        allocation_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # RESOURCE SHORTAGE ANALYSIS
    # ========================================================

    st.header("⚠️ Resource Shortage Analysis")

    shortage_food = max(
        food_required - allocated_food,
        0
    )

    shortage_water = max(
        water_required - allocated_water,
        0
    )

    shortage_medicine = max(
        medical_supplies_required - allocated_medicine,
        0
    )

    shortage_rescue = max(
        rescue_teams_required - allocated_rescue,
        0
    )

    shortage_shelter = max(
        shelter_required - allocated_shelter,
        0
    )

    shortage_df = pd.DataFrame({

        "Resource": [
            "Food",
            "Water",
            "Medical Supplies",
            "Rescue Teams",
            "Shelter"
        ],

        "Shortage": [
            shortage_food,
            shortage_water,
            shortage_medicine,
            shortage_rescue,
            shortage_shelter
        ]

    })

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        shortage_df["Resource"],
        shortage_df["Shortage"]
    )

    ax.set_title("Resource Shortage Analysis")
    ax.set_xlabel("Resource")
    ax.set_ylabel("Shortage Quantity")

    ax.bar_label(
        bars,
        fmt="%.0f",
        padding=3
    )

    ax.tick_params(
        axis="x",
        labelrotation=15
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


    # ========================================================
    # ML VS DEEP LEARNING COMPARISON
    # ========================================================

    st.header("🤖 Machine Learning vs 🧠 Deep Learning")

    comparison_df = pd.DataFrame({

        "Resource": [
            "Food",
            "Water",
            "Medical Supplies",
            "Rescue Teams",
            "Shelter"
        ],

        "Random Forest": [
            round(food_ml),
            round(water_ml),
            round(medicine_ml),
            round(rescue_ml),
            round(shelter_ml)
        ],

        "ANN": [
            round(food_dl),
            round(water_dl),
            round(medicine_dl),
            round(rescue_dl),
            round(shelter_dl)
        ]

    })

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # AI ASSESSMENT SUMMARY
    # ========================================================

    st.header("📋 AI Assessment Summary")

    st.write(
        f"""
        **Disaster Type:** {disaster_type}

        **Location:** {location}

        **Severity:** {disaster_severity}/10

        **Population Affected:** {population_affected:,}

        **Injuried People:** {injuried_people:,}

        **Displaced People:** {displaced_people:,}

        **Expected Duration:** {duration_days} days

        **Temperature:** {temperature} °C

        **Rainfall:** {rainfall}

        **Infrastructure Damage:** {infrastructure_damage}%

        **Priority Score:** {priority_score}/100

        **Priority Level:** {priority}
        """
    )


    # ========================================================
    # PRIORITY ALERT
    # ========================================================

    if priority == "CRITICAL":

        st.error(
            "🚨 CRITICAL PRIORITY — "
            "Immediate resource planning required."
        )

    elif priority == "HIGH":

        st.warning(
            "⚠️ HIGH PRIORITY — "
            "Significant relief resources required."
        )

    elif priority == "MEDIUM":

        st.info(
            "ℹ️ MEDIUM PRIORITY — "
            "Monitor the situation and allocate resources."
        )

    else:

        st.success(
            "✅ LOW PRIORITY — Standard relief planning."
        )


    # ========================================================
    # SHORTAGE ALERT
    # ========================================================

    shortage_values = [
        shortage_food,
        shortage_water,
        shortage_medicine,
        shortage_rescue,
        shortage_shelter
    ]

    if any(value > 0 for value in shortage_values):

        st.warning(
            "⚠️ Some predicted resource requirements "
            "cannot be fully met with the current inventory. "
            "Additional resources may be needed."
        )

    else:

        st.success(
            "✅ The current allocations cover all "
            "predicted resource requirements."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI Disaster Relief Resource Allocation System | "
    "Machine Learning + Deep Learning + "
    "Data Analysis + Matplotlib + Resource Allocation"
)