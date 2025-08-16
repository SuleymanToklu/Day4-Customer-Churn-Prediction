import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="Customer Churn Prediction", page_icon="👋", layout="wide")
texts = {
    'tr': {
        'page_title': "Müşteri Kaybı Tahmini", 'title': "👋 Müşteri Kaybı (Churn) Tahmincisi", 'tab_predict': "🧠 Tahmin Aracı",
        'tab_details': "🎯 Proje Detayları", 'lang_choice': "Dil Seçimi", 'prediction_header': "Canlı Müşteri Kaybı Tahmini",
        'prediction_subheader': "Bir müşterinin şirketi terk etme olasılığını tahmin etmek için aşağıdaki bilgileri girin.",
        'personal_info': "👤 Kişisel Bilgiler", 'gender': "Cinsiyet", 'partner': "Partneri Var mı?",
        'senior_citizen': "65 Yaş Üstü", 'senior_citizen_help': "Müşteri 65 yaşın üzerindeyse Evet, değilse Hayır seçin.",
        'dependents': "Bakmakla Yükümlü Olduğu Kişi Var mı?", 'subscription_info': "📄 Abonelik ve Fatura Bilgileri",
        'tenure': "Şirkette Geçirdiği Ay Sayısı", 'contract': "Sözleşme Türü", 'paperless_billing': "Kağıtsız Fatura",
        'payment_method': "Ödeme Yöntemi", 'monthly_charges': "Aylık Ücret ($)", 'total_charges': "Toplam Harcama ($)",
        'service_details': "💻 Servis Kullanım Detayları", 'phone_service': "Telefon Hizmeti",
        'multiple_lines': "Çoklu Hat", 'internet_service': "İnternet Servisi", 'online_security': "Online Güvenlik",
        'online_backup': "Online Yedekleme", 'device_protection': "Cihaz Koruma", 'tech_support': "Teknik Destek",
        'streaming_tv': "TV Yayını", 'streaming_movies': "Film Yayını", 'submit_button': "Terk Etme Olasılığını Tahmin Et",
        'prediction_result': "🔮 Tahmin Sonucu", 'churn_yes': "Bu müşterinin şirketi TERK ETME olasılığı yüksek!",
        'churn_no': "Bu müşterinin şirkette KALMA olasılığı yüksek!", 'churn_prob_label': "Terk Etme (Churn) Olasılığı",
        'options_yes_no': ('Evet', 'Hayır'), 'gender_options': ('Erkek', 'Kadın'),
        'contract_options': ('Aylık', 'Bir Yıllık', 'İki Yıllık'),
        'payment_method_options': ('Elektronik Çek', 'Posta Çeki', 'Banka Transferi (Otomatik)', 'Kredi Kartı (Otomatik)'),
        'multiple_lines_options': ('Evet', 'Hayır', 'Telefon hizmeti yok'),
        'internet_service_options': ('DSL', 'Fiber Optik', 'İnternet yok'),
        'internet_addon_options': ('Evet', 'Hayır', 'İnternet hizmeti yok'),
        'project_details_header': "Projenin Amacı ve Teknik Detaylar",
        'project_details_text': """...""", # Bu metni bir önceki cevaptan alabilirsin
        'model_performance_header': "📊 Model Performansı",
        'accuracy_metric_label': "🎯 Model Doğruluğu (Accuracy)",
        'accuracy_metric_help': "Modelin test verisindeki genel doğruluk oranıdır.",
        'f1_metric_label': "⚖️ F1 Skoru (Churn=Yes için)",
        'f1_metric_help': "Pozitif sınıfın (Churn=Yes) precision ve recall değerlerinin harmonik ortalamasıdır.",
        'ci_label': "**%95 Güven Aralığı:**",
        'classification_report_header': "Sınıflandırma Raporu (Classification Report)",
        'classification_report_subheader': "Her sınıf için Precision, Recall ve F1-Skoru gibi detaylı metrikleri gösterir.",
        'metrics_expander_header': "ℹ️ Bu metrikler ne anlama geliyor?",
        'metrics_expander_content': """
        - **Accuracy (Doğruluk):** Tüm tahminler içinde doğru olanların yüzdesi. Genel bir ölçüttür.
        - **Precision (Kesinlik):** Modelin "Terk Edecek" dediği müşterilerin gerçekten ne kadarının terk ettiğini gösterir. Yanlış pozitifleri (false positive) minimize etmek istediğimizde önemlidir.
        - **Recall (Duyarlılık):** Gerçekten terk eden müşterilerin ne kadarını doğru tespit edebildiğimizi gösterir. Pozitif vakaları (terk edenleri) kaçırmamak istediğimizde önemlidir.
        - **F1-Skoru:** Precision ve Recall'un harmonik ortalamasıdır. Bu iki metrik arasında bir denge kurar.
        - **Güven Aralığı (Confidence Interval):** Model performansının şans eseri olmadığını ve büyük olasılıkla bu aralıkta bir değere sahip olduğunu gösteren istatistiksel bir ölçümdür. Aralığın dar olması, sonucun daha istikrarlı ve güvenilir olduğunu gösterir.
        """
    },
    'en': {
        'page_title': "Customer Churn Prediction", 'title': "👋 Customer Churn Predictor", 'tab_predict': "🧠 Prediction Tool",
        'tab_details': "🎯 Project Details", 'lang_choice': "Language Selection", 'prediction_header': "Live Customer Churn Prediction",
        'prediction_subheader': "Enter the following information to predict the likelihood of a customer churning.",
        'personal_info': "👤 Personal Information", 'gender': "Gender", 'partner': "Has Partner?",
        'senior_citizen': "Senior Citizen (Over 65)", 'senior_citizen_help': "Select Yes if the customer is over 65, otherwise No.",
        'dependents': "Has Dependents?", 'subscription_info': "📄 Subscription and Billing Information",
        'tenure': "Tenure in Months", 'contract': "Contract Type", 'paperless_billing': "Paperless Billing",
        'payment_method': "Payment Method", 'monthly_charges': "Monthly Charges ($)", 'total_charges': "Total Charges ($)",
        'service_details': "💻 Service Usage Details", 'phone_service': "Phone Service",
        'multiple_lines': "Multiple Lines", 'internet_service': "Internet Service", 'online_security': "Online Security",
        'online_backup': "Online Backup", 'device_protection': "Device Protection", 'tech_support': "Tech Support",
        'streaming_tv': "Streaming TV", 'streaming_movies': "Streaming Movies", 'submit_button': "Predict Churn Probability",
        'prediction_result': "🔮 Prediction Result", 'churn_yes': "This customer is LIKELY to CHURN!",
        'churn_no': "This customer is LIKELY to STAY!", 'churn_prob_label': "Churn Probability",
        'options_yes_no': ('Yes', 'No'), 'gender_options': ('Male', 'Female'),
        'contract_options': ('Month-to-month', 'One year', 'Two year'),
        'payment_method_options': ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'),
        'multiple_lines_options': ('Yes', 'No', 'No phone service'),
        'internet_service_options': ('DSL', 'Fiber optic', 'No'),
        'internet_addon_options': ('Yes', 'No', 'No internet service'),
        'project_details_header': "Project Purpose and Technical Details",
        'project_details_text': """...""", # You can get this text from the previous response
        'model_performance_header': "📊 Model Performance",
        'accuracy_metric_label': "🎯 Model Accuracy",
        'accuracy_metric_help': "The overall accuracy of the model on the test data.",
        'f1_metric_label': "⚖️ F1 Score (for Churn=Yes)",
        'f1_metric_help': "The harmonic mean of precision and recall for the positive class (Churn=Yes). Important for imbalanced datasets.",
        'ci_label': "**95% Confidence Interval:**",
        'classification_report_header': "Classification Report",
        'classification_report_subheader': "Shows detailed metrics like Precision, Recall, and F1-Score for each class.",
        'metrics_expander_header': "ℹ️ What do these metrics mean?",
        'metrics_expander_content': """
        - **Accuracy:** The percentage of correct predictions out of all predictions. A general measure.
        - **Precision:** Of all the customers the model predicted would churn, what percentage actually churned. Important when minimizing false positives is the goal.
        - **Recall:** Of all the customers that actually churned, what percentage did the model correctly identify. Important when not missing positive cases (churners) is the goal.
        - **F1-Score:** The harmonic mean of Precision and Recall. It provides a balance between these two metrics.
        - **Confidence Interval:** A statistical measure indicating that the model's performance is not by chance and likely has a value within this range. A narrow interval indicates a more stable and reliable result.
        """
    }
}
TR_TO_EN_MAP = {
    'Evet': 'Yes', 'Hayır': 'No', 'Erkek': 'Male', 'Kadın': 'Female',
    'Aylık': 'Month-to-month', 'Bir Yıllık': 'One year', 'İki Yıllık': 'Two year',
    'Elektronik Çek': 'Electronic check', 'Posta Çeki': 'Mailed check', 
    'Banka Transferi (Otomatik)': 'Bank transfer (automatic)', 'Kredi Kartı (Otomatik)': 'Credit card (automatic)',
    'Telefon hizmeti yok': 'No phone service', 'Fiber Optik': 'Fiber optic', 'İnternet yok': 'No',
    'İnternet hizmeti yok': 'No internet service'
}


@st.cache_resource
def load_resources():
    try:
        model = joblib.load('model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        encoders = joblib.load('encoders.pkl')
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
        return model, model_columns, encoders, metrics
    except FileNotFoundError:
        st.error("Gerekli model dosyaları (model.pkl, encoders.pkl, metrics.json) bulunamadı. Lütfen önce `train_model.py`'yi çalıştırın.")
        st.stop()

model, model_columns, encoders, metrics = load_resources()

if 'language' not in st.session_state:
    st.title("Welcome / Hoş Geldiniz")
    st.write("Please select your language to continue / Devam etmek için lütfen dilinizi seçin")
    col1, col2 = st.columns(2)
    if col1.button("English 🇬🇧", use_container_width=True):
        st.session_state.language = 'en'; st.rerun()
    if col2.button("Türkçe 🇹🇷", use_container_width=True):
        st.session_state.language = 'tr'; st.rerun()
    st.stop()

with st.sidebar:
    lang_code = st.session_state.language; st.header(texts[lang_code]['lang_choice'])
    current_lang_index = ['en', 'tr'].index(lang_code)
    selected_language = st.radio("Language / Dil", ['en', 'tr'], index=current_lang_index,
        format_func=lambda x: "English 🇬🇧" if x == 'en' else "Türkçe 🇹🇷", label_visibility="collapsed")
    if lang_code != selected_language:
        st.session_state.language = selected_language; st.rerun()
lang = st.session_state.language

st.title(texts[lang]['title'])
tab1, tab2 = st.tabs([texts[lang]['tab_predict'], texts[lang]['tab_details']])

with tab1:
    st.header(texts[lang]['prediction_header']); st.write(texts[lang]['prediction_subheader'])
    with st.form(key='prediction_form'):
        with st.expander(texts[lang]['personal_info'], expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                gender_display = st.selectbox(texts[lang]['gender'], texts[lang]['gender_options'])
                partner_display = st.selectbox(texts[lang]['partner'], texts[lang]['options_yes_no'])
            with col2:
                senior_citizen_display = st.selectbox(texts[lang]['senior_citizen'], texts[lang]['options_yes_no'], help=texts[lang]['senior_citizen_help'])
                dependents_display = st.selectbox(texts[lang]['dependents'], texts[lang]['options_yes_no'])
        with st.expander(texts[lang]['subscription_info']):
            col1, col2, col3 = st.columns(3)
            with col1:
                tenure = st.slider(texts[lang]['tenure'], 0, 72, 24)
                contract_display = st.selectbox(texts[lang]['contract'], texts[lang]['contract_options'])
            with col2:
                paperless_billing_display = st.selectbox(texts[lang]['paperless_billing'], texts[lang]['options_yes_no'])
                payment_method_display = st.selectbox(texts[lang]['payment_method'], texts[lang]['payment_method_options'])
            with col3:
                monthly_charges = st.slider(texts[lang]['monthly_charges'], 18.0, 118.0, 70.0)
                total_charges = st.slider(texts[lang]['total_charges'], 18.0, 8700.0, 1400.0)
        with st.expander(texts[lang]['service_details']):
            col1, col2, col3 = st.columns(3)
            with col1:
                phone_service_display = st.selectbox(texts[lang]['phone_service'], texts[lang]['options_yes_no'])
                multiple_lines_display = st.selectbox(texts[lang]['multiple_lines'], texts[lang]['multiple_lines_options'])
                internet_service_display = st.selectbox(texts[lang]['internet_service'], texts[lang]['internet_service_options'])
            with col2:
                online_security_display = st.selectbox(texts[lang]['online_security'], texts[lang]['internet_addon_options'])
                online_backup_display = st.selectbox(texts[lang]['online_backup'], texts[lang]['internet_addon_options'])
                device_protection_display = st.selectbox(texts[lang]['device_protection'], texts[lang]['internet_addon_options'])
            with col3:
                tech_support_display = st.selectbox(texts[lang]['tech_support'], texts[lang]['internet_addon_options'])
                streaming_tv_display = st.selectbox(texts[lang]['streaming_tv'], texts[lang]['internet_addon_options'])
                streaming_movies_display = st.selectbox(texts[lang]['streaming_movies'], texts[lang]['internet_addon_options'])
        st.markdown("---"); submit_button = st.form_submit_button(label=texts[lang]['submit_button'])

    if submit_button:
        senior_citizen = 1 if senior_citizen_display == texts[lang]['options_yes_no'][0] else 0

        if lang == 'tr':
            gender = TR_TO_EN_MAP.get(gender_display, gender_display)
            partner = TR_TO_EN_MAP.get(partner_display, partner_display)
            dependents = TR_TO_EN_MAP.get(dependents_display, dependents_display)
            paperless_billing = TR_TO_EN_MAP.get(paperless_billing_display, paperless_billing_display)
            phone_service = TR_TO_EN_MAP.get(phone_service_display, phone_service_display)
            contract = TR_TO_EN_MAP.get(contract_display, contract_display)
            payment_method = TR_TO_EN_MAP.get(payment_method_display, payment_method_display)
            multiple_lines = TR_TO_EN_MAP.get(multiple_lines_display, multiple_lines_display)
            internet_service = TR_TO_EN_MAP.get(internet_service_display, internet_service_display)
            online_security = TR_TO_EN_MAP.get(online_security_display, online_security_display)
            online_backup = TR_TO_EN_MAP.get(online_backup_display, online_backup_display)
            device_protection = TR_TO_EN_MAP.get(device_protection_display, device_protection_display)
            tech_support = TR_TO_EN_MAP.get(tech_support_display, tech_support_display)
            streaming_tv = TR_TO_EN_MAP.get(streaming_tv_display, streaming_tv_display)
            streaming_movies = TR_TO_EN_MAP.get(streaming_movies_display, streaming_movies_display)
        else: # Dil İngilizce ise, değerleri doğrudan al
            gender, partner, dependents, paperless_billing, phone_service, contract, payment_method, multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies = (
                gender_display, partner_display, dependents_display, paperless_billing_display, phone_service_display, contract_display, payment_method_display, multiple_lines_display, internet_service_display, online_security_display, online_backup_display, device_protection_display, tech_support_display, streaming_tv_display, streaming_movies_display
            )

        input_data = {
            'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines, 'InternetService': internet_service,
            'OnlineSecurity': online_security, 'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
            'TechSupport': tech_support, 'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
            'Contract': contract, 'PaperlessBilling': paperless_billing, 'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
        }
        input_df = pd.DataFrame([input_data])
        
        for col, le in encoders.items():
            if col in input_df.columns and input_df[col].dtype == 'object':
                input_df[col] = le.transform(input_df[col])
        
        input_df = input_df[model_columns]; prediction = model.predict(input_df); prediction_proba = model.predict_proba(input_df)
        st.subheader(texts[lang]['prediction_result']); churn_probability = prediction_proba[0][1]
        
        if prediction[0] == 1: st.error(texts[lang]['churn_yes'])
        else: st.success(texts[lang]['churn_no'])
        
        st.metric(label=texts[lang]['churn_prob_label'], value=f"{churn_probability:.2%}"); st.progress(float(churn_probability))

with tab2:
    st.header(texts[lang]['project_details_header'])
    st.write(texts[lang]['project_details_text'].strip())
    
    st.markdown("---")
    st.header(texts[lang]['model_performance_header']) 
    
    acc = metrics['accuracy']
    acc_ci = metrics['accuracy_confidence_interval']
    f1 = metrics['f1_score']
    f1_ci = metrics['f1_score_confidence_interval']
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label=texts[lang]['accuracy_metric_label'], 
            value=f"{acc:.2%}",
            help=texts[lang]['accuracy_metric_help'] 
        )
        st.write(f"{texts[lang]['ci_label']} `{acc_ci[0]:.2%} - {acc_ci[1]:.2%}`")

    with col2:
        st.metric(
            label=texts[lang]['f1_metric_label'], 
            value=f"{f1:.2f}",
            help=texts[lang]['f1_metric_help'] 
        )
        st.write(f"{texts[lang]['ci_label']} `{f1_ci[0]:.2f} - {f1_ci[1]:.2f}`") 

    st.markdown(f"##### {texts[lang]['classification_report_header']}") 
    st.write(texts[lang]['classification_report_subheader']) 
    
    report_df = pd.DataFrame(metrics['classification_report']).transpose()
    st.dataframe(report_df.round(2))

    with st.expander(texts[lang]['metrics_expander_header']): 
        st.write(texts[lang]['metrics_expander_content']) 