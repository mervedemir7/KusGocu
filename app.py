import streamlit as st
import pandas as pd
import pydeck as pdk
import io
import numpy as np
import joblib

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Kuş Göç Yolları Analizi ve Tahmini",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': 'Bu uygulama, kuş göç verilerini etkileşimli bir harita üzerinde görselleştirmek ve makine öğrenimi modeli kullanarak göç başarısını tahmin etmek için geliştirilmiştir. Shneiderman\'ın kullanıcı arayüzü tasarım ilkeleri referakm alınmıştır.',
        'Get help': 'https://github.com/mervedemir7/bird-migration-analysis/issues',
        'Report a bug': 'https://github.com/mervedemir7/bird-migration-analysis/issues'
    }
)

# Tema renkleri - Orman Teması
FOREST_BACKGROUND = "#F5F5DC"  # Kremsi, hafif bej
FOREST_PRIMARY = "#556B2F"      # Koyu Zeytin Yeşili (Ana yazı rengi ve bazı elementler için)
FOREST_ACCENT_GREEN = "#8FBC8F" # Açık Orman Yeşili (Arka plan ve vurgular)
FOREST_ACCENT_BROWN = "#A0522D" # Terrakota kahverengi (Butonlar ve vurgular)
FOREST_DARK = "#2F4F4F"         # Çok koyu yeşil/gri (Başlıklar ve önemli yazılar)

# Estetik CSS
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Times+New_Roman&display=swap');

    html, body, [class*="st-emotion"], .stApp {{
        font-family: 'Times New Roman', serif;
        background-color: {FOREST_BACKGROUND};
        color: {FOREST_PRIMARY}; /* Ana yazı rengi */
    }}
    .stApp {{
        background-color: {FOREST_BACKGROUND};
        color: {FOREST_PRIMARY};
    }}
    .css-1d3joqt {{ /* Sidebar background */
        background-color: {FOREST_ACCENT_GREEN};
        color: {FOREST_DARK};
    }}
    .main .block-container {{
        background-color: {FOREST_BACKGROUND};
        padding: 1rem;
    }}
    h1 {{
        font-size: 2.8rem;
        color: {FOREST_DARK}; /* Koyu başlıklar */
        text-align: center;
        padding-bottom: 0.5rem;
        font-family: 'Times New Roman', serif;
    }}
    h2, h3, h4, h5, h6 {{
        color: {FOREST_ACCENT_BROWN}; /* Vurgulu başlıklar */
        font-family: 'Times New Roman', serif;
    }}
    .stSelectbox, .stMultiSelect {{
        margin-bottom: 15px;
    }}
    .stButton>button {{
        background-color: {FOREST_ACCENT_BROWN};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s ease;
        font-family: 'Times New Roman', serif;
    }}
    .stButton>button:hover {{
        background-color: #B25F3D; /* Terrakota renginin biraz koyusu */
    }}
    .stSpinner > div > div {{
        border-top-color: {FOREST_ACCENT_BROWN} !important;
    }}
    .stAlert {{
        border-radius: 8px;
        padding: 1rem;
        background-color: rgba(143, 188, 143, 0.2); /* Hafif yeşil tonlu uyarılar */
        color: {FOREST_DARK};
        font-family: 'Times New Roman', serif;
    }}
    .stMultiSelect label, .stSelectbox label, .stSlider label, .stTextInput label {{
        font-weight: bold;
        color: {FOREST_DARK}; /* Koyu etiketler */
        font-family: 'Times New Roman', serif;
    }}
    .stSelectbox .css-1wa3qg0, .stMultiSelect .css-1wa3qg0 {{ /* Input box text color */
        color: {FOREST_PRIMARY};
    }}
    .stMarkdown {{
        color: {FOREST_PRIMARY};
        font-family: 'Times New Roman', serif;
    }}
    </style>
    """, unsafe_allow_html=True)

# Veri işleme fonksiyonu (harita için)
@st.cache_data
def process_data(df_input):
    with st.spinner("Veriler işleniyor ve harita için hazırlanıyor..."):
        df = df_input.copy()

        # Harita için gerekli sütunların kontrolü
        required_coords = ["Start_Latitude", "Start_Longitude", "End_Latitude", "End_Longitude", "Flight_Distance_km"]
        if not all(col in df.columns for col in required_coords):
            st.error("Harita oluşturmak için gerekli koordinat ve mesafe sütunları (Başlangıç Enlem, Başlangıç Boylam, Bitiş Enlem, Bitiş Boylam, Uçuş Mesafesi) yüklenen dosyada bulunamadı. Lütfen dosyanızı kontrol edin.")
            return pd.DataFrame() # Boş DataFrame döndür

        for col in ["Start_Latitude", "Start_Longitude", "End_Latitude", "End_Longitude", "Flight_Distance_km"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=["Start_Latitude", "Start_Longitude", "End_Latitude", "End_Longitude"], inplace=True)

        # Harita filtrelemesi ve tooltip için Türkçe çeviriler
        # Eğer bu sütunlar yoksa 'Bilinmiyor' atayalım, böylece hata vermez.
        df['Migration_Start_Month_TR'] = df.get('Migration_Start_Month', 'Unknown').astype(str).apply(lambda x: x.split(',')[0].strip()).map({
            "Jan": "Ocak", "Feb": "Şubat", "Mar": "Mart", "Apr": "Nisan", "May": "Mayıs", "Jun": "Haziran",
            "Jul": "Temmuz", "Aug": "Ağustos", "Sep": "Eylül", "Oct": "Ekim", "Nov": "Kasım", "Dec": "Aralık"
        }).fillna("Bilinmeyen Ay")
        month_order_tr = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
                          "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
        df["Migration_Start_Month_TR"] = pd.Categorical(df["Migration_Start_Month_TR"], categories=month_order_tr, ordered=True)

        df['Species_TR'] = df.get('Species', 'Unknown').map({
            "Stork": "Leylek", "Warder": "Balıkçı", "Crane": "Turna", "Hawk": "Şahin",
            "Goose": "Kaz", "Eagle": "Kartal", "Owl": "Baykuş", "Robin": "Kızılgerdan",
            "Sparrow": "Serçe", "Seagull": "Martı", "Pelican": "Pelikan", "Duck": "Ördek",
            "Pigeon": "Güvercin", "Crow": "Karga", "Penguin": "Penguen", "Ostrich": "Devekuşu",
            "Kiwi": "Kivi", "Parrot": "Papağan", "Flamingo": "Flamingo", "Swan": "Kuğu",
            "Falcon": "Atmaca", "Vulture": "Akbaba", "Hummingbird": "Sinekkuşu",
            "Woodpecker": "Ağaçkakan", "Kingfisher": "Yalıçapkını", "Osprey": "Balık Kartalı",
            "Albatross": "Albatros", "Condor": "Kondor", "Macaw": "Ara", "Canary": "Kanarya",
            "Quail": "Bıldırcın", "Raven": "Kuzgun", "Rooster": "Horoz", "Sandpiper": "Kumkuşu",
            "Skylark": "Tarlakuşu", "Starling": "Sığırcık", "Swallow": "Kırlangıç", "Toucan": "Tukan",
            "Turkey": "Hindi", "Wren": "Çitkuşu", "Puffin": "Deniz Papağanı", "Cormorant": "Karabatak",
            "Heron": "Balıkçıl", "Ibis": "İbis", "Plover": "Kıyı Kuşu", "Teal": "Çamurcun",
            "Wagtail": "Kuyruksallayan", "Woodcock": "Çulluk", "Pipit": "İncirkuşu",
            "Other": "Diğer", "Unknown": "Bilinmeyen"
        }).fillna("Bilinmeyen Tür")

        df['Region_TR'] = df.get('Region', 'Unknown').map({
            "North America": "Kuzey Amerika", "South America": "Güney Amerika", "Europe": "Avrupa",
            "Asia": "Asya", "Africa": "Afrika", "Oceania": "Okyanusya", "Arctic": "Arktik",
            "Antarctic": "Antarktika", "Grassland": "Otlak", "Forest": "Orman", "Urban": "Kentsel",
            "Coastal": "Kıyı", "Wetland": "Sulak Alan", "Mountain": "Dağlık Bölge", "Desert": "Çöl",
            "Tundra": "Tundra", "Tropikal": "Tropikal", "Temperate": "Ilıman", "Polar": "Kutup",
            "Continental": "Kıtasal", "Ada": "Ada", "Denizel": "Denizel", "Riverine": "Nehir Kıyısı",
            "Savanna": "Savana", "Steppe": "Step", "Taiga": "Tayga", "Subtropical": "Subtropikal",
            "Mediterranean": "Akdeniz", "Boreal": "Boreal", "Alpine": "Alpin",
            "Other": "Diğer", "Unknown": "Bilinmeyen"
        }).fillna("Bilinmeyen Bölge")

        df['Migration_Reason_TR'] = df.get('Migration_Reason', 'Unknown').map({
            "Feeding": "Beslenme", "Breeding": "Üreme", "Climate": "İklim Koşulları",
            "Shelter": "Barınma", "Predator Avoidance": "Avcıdan Kaçınma",
            "Climate Change": "İlim Değişikliği", "Resource Scarcity": "Kaynak Kıtlığı",
            "Nesting Site": "Yuvalama Alanı", "Safety": "Güvenlik",
            "Seasonal Change": "Mevsimsel Değişim", "Food Availability": "Yiyecek Bulunabilirliği",
            "Water Availability": "Su Bulunabilirliği", "Habitat Loss": "Habitat Kaybı",
            "Other": "Diğer", "Unknown": "Bilinmeyen"
        }).fillna("Bilinmeyen Neden")

        df["Start_Coords_TR"] = df.apply(lambda row: f"Enlem: {row['Start_Latitude']:.2f}, Boylam: {row['Start_Longitude']:.2f}", axis=1)
        df["End_Coords_TR"] = df.apply(lambda row: f"Enlem: {row['End_Latitude']:.2f}, Boylam: {row['End_Longitude']:.2f}", axis=1)

        return df

# --- Makine Öğrenimi Pipeline'ını Yükleme ---
@st.cache_resource
def load_prediction_pipeline():
    try:
        pipeline = joblib.load('logistic_model_9_features_5k_samples.pkl')
        return pipeline
    except FileNotFoundError:
        st.error("❗ **Hata:** 'logistic_model_9_features_5k_samples.pkl' model dosyası bulunamadı. Lütfen model dosyasını uygulamanın aynı dizinine yüklediğinizden emin olun.")
        st.stop()
    except Exception as e:
        st.error(f"❗ **Hata:** Model Pipeline'ı yüklenirken beklenmeyen bir sorun oluştu: {e}")
        st.stop()

prediction_pipeline = load_prediction_pipeline()

# --- Ana Uygulama Akışı ---
st.title("🦅 Kuş Göç Yolları Analizi ve Tahmini")
st.markdown("""
Bu interaktif uygulama, kuş göç verilerini anlaşılır ve etkileşimli bir harita üzerinde görselleştirmenizi sağlar ve makine öğrenimi modeli kullanarak göç başarısını tahmin eder.
Sol paneldeki filtreleri kullanarak göç rotalarını tür, bölge ve başlangıç ayına göre inceleyebilirsiniz.
""")

st.info("💡 **İpucu:** Daha detaylı analiz için sol panelden kendi CSV dosyanızı yükleyebilir veya filtreleri kullanarak haritayı sadeleştirebilirsiniz.")

# --- SIDEBAR (Sol Panel) ---
st.sidebar.header("📊 Veri Yükleme")
uploaded_file = st.sidebar.file_uploader(
    "Kuş Göçü Verilerini Yükle (.csv)",
    type=["csv"],
    help="Lütfen 'Species', 'Region', 'Migration_Start_Month', 'Migration_Reason', 'Start_Latitude', 'Start_Longitude', 'End_Latitude', 'End_Longitude', 'Flight_Distance_km' sütunlarını içeren bir CSV dosyası yükleyin."
)

df_map = pd.DataFrame() # Harita için kullanılacak DataFrame

if uploaded_file is None:
    st.info("Devam etmek için lütfen sol panelden bir CSV dosyası yükleyin (Harita ve Tablo Görünümü için).")
    # Eğer dosya yüklenmediyse, varsayılan olarak boş bir DataFrame oluşturup,
    # sadece tahmin bölümü için varsayılan seçenekleri kullanacağız.
    # Bu, uygulamanın harita kısmı için veri olmadan da çalışmasını sağlar.
    df_dummy = pd.DataFrame({
        'Weather_Condition': ['Güneşli'], 'Pressure_hPa': [1010.0], 'Migration_Start_Month': ['Nisan'],
        'Species': ['Leylek'], 'Region': ['Avrupa'], 'Migrated_in_Flock': ['Evet'],
        'Flight_Distance_km': [2500.0], 'Temperature_C': [15.0], 'Wind_Speed_kmh': [10.0]
    })
    # df_map = df_dummy # Varsayılan veri ile başlamak isterseniz bu satırı yorumdan çıkarabilirsiniz.
else:
    try:
        df_raw = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
        df_map = process_data(df_raw)

        if df_map.empty:
            st.sidebar.error("Yüklenen CSV dosyasında işlenebilecek veri bulunamadı veya işleme sırasında hata oluştu.")
        else:
            st.sidebar.success("Veri başarıyla yüklendi ve işlendi!")

    except pd.errors.EmptyDataError:
        st.sidebar.error("❗ **Hata:** Yüklenen CSV dosyası boş.")
    except pd.errors.ParserError:
        st.sidebar.error("❗ **Hata:** Yüklenen CSV dosyası okunamıyor. Lütfen dosya formatını veya içeriğini kontrol edin.")
    except Exception as e:
        st.sidebar.error(f"❗ **Beklenmeyen bir hata oluştu**: Dosya yüklenirken sorun yaşandı. Hata detayları: `{e}`")


# --- HARİTA VE FİLTRELEME BÖLÜMÜ ---
# Harita bölümünü, df_map boş değilse gösteriyoruz.
if not df_map.empty and all(col in df_map.columns for col in ["Start_Latitude", "Start_Longitude", "End_Latitude", "End_Longitude"]):
    st.sidebar.header("🗺️ Harita Görselleştirme Filtreleri")
    st.sidebar.markdown("Göç verilerini harita üzerinde detaylandırmak için aşağıdaki filtreleri kullanın.")

    # Kuş Türü Seçimi
    st.sidebar.subheader("Kuş Türü")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        select_all_species = st.button("Tümünü Seç", key="all_species_btn_map")
    with col2:
        clear_species = st.button("Seçimi Temizle", key="clear_species_btn_map")

    all_species_tr = sorted(df_map["Species_TR"].unique())
    if "species_selected_tr" not in st.session_state:
        st.session_state.species_selected_tr = []

    if select_all_species:
        st.session_state.species_selected_tr = all_species_tr
    elif clear_species:
        st.session_state.species_selected_tr = []

    species_selected_tr = st.sidebar.multiselect(
        "Kuş Türü",
        all_species_tr,
        default=st.session_state.species_selected_tr,
        help="Görselleştirilecek kuş türlerini seçin. Birden fazla seçim yapabilirsiniz."
    )
    if species_selected_tr != st.session_state.species_selected_tr:
        st.session_state.species_selected_tr = species_selected_tr

    st.sidebar.markdown("---") # Ayırıcı ekle

    # Bölge Seçimi
    st.sidebar.subheader("Bölge")
    col3, col4 = st.sidebar.columns(2)
    with col3:
        select_all_regions = st.button("Tümünü Seç", key="all_regions_btn_map")
    with col4:
        clear_regions = st.button("Seçimi Temizle", key="clear_regions_btn_map")

    all_regions_tr = sorted(df_map["Region_TR"].unique())
    if "region_selected_tr" not in st.session_state:
        st.session_state.region_selected_tr = []

    if select_all_regions:
        st.session_state.region_selected_tr = all_regions_tr
    elif clear_regions:
        st.session_state.region_selected_tr = []

    region_selected_tr = st.sidebar.multiselect(
        "Bölge",
        all_regions_tr,
        default=st.session_state.region_selected_tr,
        help="Kuşların gözlemlendiği coğrafi bölgeleri seçin."
    )
    if region_selected_tr != st.session_state.region_selected_tr:
        st.session_state.region_selected_tr = region_selected_tr

    st.sidebar.markdown("---") # Ayırıcı ekle

    # Göç Başlangıç Ayı Seçimi
    st.sidebar.subheader("Göç Başlangıç Ayı")
    all_months_tr = df_map["Migration_Start_Month_TR"].cat.categories.tolist() if "Migration_Start_Month_TR" in df_map.columns else []
    default_month_index = all_months_tr.index(df_map["Migration_Start_Month_TR"].mode()[0]) if not df_map["Migration_Start_Month_TR"].empty and "Migration_Start_Month_TR" in df_map.columns else 0

    start_month_tr = st.sidebar.selectbox(
        "Göç Başlangıç Ayı",
        all_months_tr,
        index=default_month_index,
        help="Göç hareketini gözlemlemek istediğiniz başlangıç ayını seçin. Tek bir ay seçimi haritayı sadeleştirebilir."
    )

    filtered_map_data = df_map.copy()
    if species_selected_tr:
        filtered_map_data = filtered_map_data[filtered_map_data["Species_TR"].isin(species_selected_tr)]
    if region_selected_tr:
        filtered_map_data = filtered_map_data[filtered_map_data["Region_TR"].isin(region_selected_tr)]
    if start_month_tr:
        filtered_map_data = filtered_map_data[filtered_map_data["Migration_Start_Month_TR"] == start_month_tr]

    st.markdown(f"**📈 Gösterilen Toplam Göç Kaydı:** `{len(filtered_map_data)}`")

    if filtered_map_data.empty:
        st.warning("⚠️ **Uyarı:** Seçilen filtre kriterlerine göre hiç göç kaydı bulunamadı. Lütfen filtrelerinizi değiştirerek daha geniş bir seçim yapmayı deneyin.")
        st.info("İpucu: Daha fazla sonuç görmek için 'Kuş Türü' veya 'Bölge' filtrelerindeki tüm seçenekleri işaretleyebilirsiniz.")
    else:
        # Harita merkezini sadece geçerli koordinatlar varsa ayarla
        if not filtered_map_data[["Start_Latitude", "Start_Longitude"]].isnull().all().all():
            view_state = pdk.ViewState(
                latitude=filtered_map_data["Start_Latitude"].mean(),
                longitude=filtered_map_data["Start_Longitude"].mean(),
                zoom=1.5,
                pitch=0,
                bearing=0
            )
        else:
            view_state = pdk.ViewState(latitude=0, longitude=0, zoom=1) # Varsayılan konum

        arc_layer = pdk.Layer(
            "ArcLayer",
            data=filtered_map_data,
            get_source_position=["Start_Longitude", "Start_Latitude"],
            get_target_position=["End_Longitude", "End_Latitude"],
            get_source_color=[0, 150, 0, 160], # Kaynak rengi: Koyu yeşil
            get_target_color=[160, 82, 45, 160], # Hedef rengi: Kahverengi
            auto_highlight=True,
            width_scale=0.0001,
            width_min_pixels=1,
            get_width="Flight_Distance_km",
            pickable=True
        )

        st.pydeck_chart(pdk.Deck(
            layers=[arc_layer],
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/satellite-v9", # Google Earth benzeri uydu görüntüsü
            tooltip={
                "html": (
                    "<b>Tür:</b> {Species_TR}<br/>"
                    "<b>Mesafe:</b> {Flight_Distance_km} km<br/>"
                    "<b>Neden:</b> {Migration_Reason_TR}<br/>"
                    "<b>Başlangıç Konumu:</b> {Start_Coords_TR}<br/>"
                    "<b>Hedef Konumu:</b> {End_Coords_TR}"
                )
            }
        ))

        st.subheader("Tablo Görünümü")
        st.markdown("Haritada gösterilen göç verilerinin detaylı listesi:")
        st.dataframe(filtered_map_data.head(100), use_container_width=True)
else:
    st.warning("Harita ve tablo görünümü için geçerli kuş göçü verisi yüklenemedi. Lütfen geçerli bir CSV dosyası yükleyin.")


# --- TAHMİN MODELİ BÖLÜMÜ ---
st.sidebar.header("🔮 Göç Başarısı Tahmini")
st.sidebar.markdown("Belirli koşullar altında kuş göçünün başarılı olup olmayacağını tahmin edin.")

st.sidebar.write("Lütfen tahmin için aşağıdaki değerleri girin:")

# Tahmin için Streamlit inputları
weather_options = ['Güneşli', 'Bulutlu', 'Yağmurlu', 'Rüzgarlı', 'Sisli']
species_options = ['Leylek', 'Kaz', 'Turna', 'Şahin', 'Kızılgerdan', 'Serçe', 'Kartal', 'Baykuş', 'Pelikan', 'Ördek']
regions_options = ['Avrupa', 'Asya', 'Kuzey Amerika', 'Afrika', 'Güney Amerika', 'Okyanusya']
months_options = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran', 'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']
flock_options = ['Evet', 'Hayır']

col1, col2 = st.sidebar.columns(2)

with col1:
    weather_condition_pred = st.selectbox('Hava Durumu', options=weather_options)
    species_pred = st.selectbox('Kuş Türü', options=species_options)
    migration_start_month = st.selectbox('Göç Ayı', options=months_options)
    temperature_c = st.slider('Sıcaklık (°C)', min_value=-15.0, max_value=40.0, value=15.0, step=0.5)

with col2:
    pressure_hpa = st.slider('Basınç (hPa)', min_value=980.0, max_value=1040.0, value=1010.0, step=0.1)
    region_pred = st.selectbox('Kıta/Bölge', options=regions_options)
    migrated_in_flock_pred = st.selectbox('Sürü Halinde Göç?', options=flock_options)
    flight_distance_km = st.number_input('Uçuş Mesafesi (km)', min_value=500, max_value=8000, value=2500, step=100)
    wind_speed_kmh = st.slider('Rüzgar Hızı (km/s)', min_value=0.0, max_value=80.0, value=20.0, step=0.5)


# MODELİN BEKLEDİĞİ 9 SÜTUNUN SIRASI
input_data_columns = [
    'Weather_Condition', 'Pressure_hPa', 'Migration_Start_Month',
    'Species', 'Region', 'Migrated_in_Flock', 'Flight_Distance_km',
    'Temperature_C', 'Wind_Speed_kmh'
]

# Girdi değerlerini modelin anlayacağı İngilizce formatına dönüştürme
weather_condition_map = {'Güneşli': 'Sunny', 'Bulutlu': 'Cloudy', 'Yağmurlu': 'Rainy', 'Rüzgarlı': 'Windy', 'Sisli': 'Foggy'}
species_map = {
    'Leylek': 'Stork', 'Kaz': 'Goose', 'Turna': 'Crane', 'Şahin': 'Hawk',
    'Kızılgerdan': 'Robin', 'Serçe': 'Sparrow', 'Kartal': 'Eagle', 'Baykuş': 'Owl',
    'Pelikan': 'Pelican', 'Ördek': 'Duck'
}
region_map = {
    "Kuzey Amerika": "North America", "Güney Amerika": "South America", "Avrupa": "Europe",
    "Asya": "Asia", "Afrika": "Africa", "Okyanusya": "Oceania", "Arktik": "Arctic",
    "Antarktika": "Antarctic", "Otlak": "Grassland", "Orman": "Forest", "Kentsel": "Urban",
    "Kıyı": "Coastal", "Sulak Alan": "Wetland", "Dağlık Bölge": "Mountain", "Çöl": "Desert",
    "Tundra": "Tundra", "Tropikal": "Tropical", "Ilıman": "Temperate", "Kutup": "Polar",
    "Kıtasal": "Continental", "Ada": "Island", "Denizel": "Marine", "Nehir Kıyısı": "Riverine",
    "Savana": "Savanna", "Steppe": "Steppe", "Tayga": "Taiga", "Subtropikal": "Subtropical",
    "Akdeniz": "Mediterranean", "Boreal": "Boreal", "Alpine": "Alpine",
    "Diğer": "Other", "Bilinmeyen": "Unknown"
}
months_map = {
    "Ocak": "Jan", "Şubat": "Feb", "Mart": "Mar", "Nisan": "Apr", "Mayıs": "May", "Haziran": "Jun",
    "Temmuz": "Jul", "Ağustos": "Aug", "Eylül": "Sep", "Ekim": "Oct", "Kasım": "Nov", "Aralık": "Dec"
}
flock_map = {'Evet': 'Yes', 'Hayır': 'No'}


input_df_for_prediction = pd.DataFrame([[
    weather_condition_map.get(weather_condition_pred, weather_condition_pred),
    pressure_hpa,
    months_map.get(migration_start_month, migration_start_month),
    species_map.get(species_pred, species_pred),
    region_map.get(region_pred, region_pred),
    flock_map.get(migrated_in_flock_pred, migrated_in_flock_pred),
    flight_distance_km,
    temperature_c,
    wind_speed_kmh
]], columns=input_data_columns)


if st.sidebar.button('Göç Başarısını Tahmin Et', key='predict_button'):
    if prediction_pipeline is not None:
        try:
            prediction = prediction_pipeline.predict(input_df_for_prediction)
            prediction_proba = prediction_pipeline.predict_proba(input_df_for_prediction)

            st.sidebar.subheader("Tahmin Sonucu:")
            if prediction[0] == 1:
                st.sidebar.success(f'✅ **Göç Başarılı Olacak!** (Olasılık: {prediction_proba[0][1]*100:.2f}%)')
            else:
                st.sidebar.error(f'❌ **Göç Başarısız Olacak!** (Olasılık: {prediction_proba[0][0]*100:.2f}%)')
            st.sidebar.info("Bu tahmin, 5000 sentetik veri örneği üzerinde eğitilmiş makine öğrenimi modeline dayanmaktadır.")
        except Exception as e:
            st.sidebar.error(f"Tahmin yapılırken bir hata oluştu: {e}")
            st.sidebar.info("Lütfen tüm giriş alanlarını doğru bir şekilde doldurduğunuzdan ve modelin beklediği tüm özellikleri sağladığınızdan emin olun.")
            st.sidebar.write(f"Hata detayı: {e}")
            st.sidebar.write("Gönderilen input DataFrame sütunları:")
            st.sidebar.write(input_df_for_prediction.columns.tolist())
            st.sidebar.write("Gönderilen input DataFrame (ilk satır):")
            st.sidebar.dataframe(input_df_for_prediction.head(1))
    else:
        st.sidebar.warning("Tahmin modeli yüklenemediği için tahmin yapılamıyor.")

# Yan Panel Alt Bilgisi
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Uygulama Bilgisi:**
Bu interaktif görselleştirme uygulaması, kuş göç yollarını kolayca keşfetmenizi ve göç başarısını tahmin etmenizi sağlar.
Veri analizini daha erişilebilir kılmak için **Shneiderman'ın Kullanıcı Arayüzü Tasarım İlkeleri** dikkate alınmıştır.
""")
st.sidebar.info("Herhangi bir soru, öneri veya geri bildiriminiz için lütfen iletişime geçmekten çekinmeyin.")