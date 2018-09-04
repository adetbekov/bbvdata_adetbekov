import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def preprocess(df, weather_df, output_data, test=False):
    print("\n🐋💨 Preproccessing started")
    print("💔 Dropping unnecessary columns in df...")
    bbvdata = df.drop(["uid", "image_name", "time", "id", "point"], axis=1)
    bbvdata = bbvdata[["amount", "bv_number", "date"]][bbvdata["amount"] > 0] if not test else bbvdata[["bv_number", "date"]]
    
    print("⚙️ Sorting df...")
    bbvdata.sort_values(["date", "bv_number"], ascending=[True, True], inplace=True)

    print("🗓 Converting string column to datetime...")
    bbvdata["date"] = pd.to_datetime(bbvdata["date"])

    print("🗓 Generating year, hour, month, chunck, day of the week columns...")
    bbvdata["Year"] = bbvdata["date"].apply(lambda d: d.year)
    bbvdata["Hour"] = bbvdata["date"].apply(lambda d: d.hour)
    bbvdata["Month"] = bbvdata["date"].apply(lambda d: d.month)
    bbvdata["Chunk"] = bbvdata["date"].apply(lambda d: int((d.hour*60+d.minute)/10))
    bbvdata["Day of the Week"] = bbvdata["date"].apply(lambda d: d.weekday()).map({0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'})

    print("✂️ Thresholding year > 2016...")
    bbvdata = bbvdata[bbvdata["Year"] > 2016]

    print("☀️ Weather preprocessing...")
    weather = weather_df[['Местное время в Алматы', 'T', 'W1', 'RRR', 'sss']][~weather_df.isnull()["T"]]
    weather_states = [
        'Туман или ледяной туман или сильная мгла.',
        'Снег или дождь со снегом.',
        'Ливень (ливни).',
        'Ливни или перемещающиеся осадки.',
        'Дождь.',
        'Осадки',
        'Явление, связанное с переносом ветром твердых частиц, видимость пониженная.',
        'Гроза (грозы) с осадками или без них.',
        'Морось.',
        'Облака покрывали более половины неба в течение всего соответствующего периода.',
        'Облака покрывали более половины неба в течение одной части соответствующего периода и половину или менее в течение другой части периода.',
        'Облака покрывали половину неба или менее в течение всего соответствующего периода.'
    ]
    weather = weather.replace(np.nan, 0)
    weather['W1'] = weather['W1'].apply(lambda w: 100-((weather_states.index(w) if type(w) != int else w) * 8))
    weather['RRR'] = weather['RRR'].replace("Следы осадков", 1.0)
    weather['sss'] = weather['sss'].replace("Менее 0.5", 0.5).replace("Снежный покров не постоянный.", 0.3)
    weather = weather[pd.to_datetime(weather['Местное время в Алматы']).dt.year > 2016]
    weather['date'] = pd.to_datetime(weather['Местное время в Алматы']).apply(lambda d: datetime(d.year, d.month, d.day, d.hour))

    print("🌧 Weather time quarter generating...")
    quarters = [0,0,0,3,3,3,6,6,6,9,9,9,12,12,12,15,15,15,18,18,18,21,21,21]
    bbvdata["date"] = bbvdata["date"].apply(lambda d: datetime(d.year, d.month, d.day, quarters[d.hour]))
    
    print("🙏🏻 Merge weather and df...")
    bbvdata = bbvdata.merge(weather.drop("Местное время в Алматы",axis=1), on='date', how='inner')
    bbvdata = bbvdata.drop(["date", "Hour", "Year"], axis=1)

    print("🔄 Sorting...")
    bbvdata.sort_values(["Month", "Day of the Week", "Chunk", "bv_number"], inplace=True)
    
    print("🔗 Generating onehot vectors...")
    X = pd.get_dummies(bbvdata, columns=["bv_number", "Day of the Week", "Month"], sparse=True)
    
    print("🚘 Generating directions...")
    X["Direction"] = bbvdata["bv_number"].apply(lambda b: 1 if b <= 8 else 0)

    print("👾 Saving preprocessed data. Please wait for a while...")
    X.to_csv(output_data)
    print("✅ Preprocessing is done.", output_data, "has been generated!")


def load_data(df_str, weather_df_str, output_data, test=False):
    print("\n📥 Loading raw", df_str, "data...")
    try:
        df = pd.read_csv(df_str, header=0)
        print("✅ Loading", df_str, "data has been successfully completed.")
    except:
        print("⛔️ Error on loading", df_str, "data!")

    print("\n📥 Loading raw", weather_df_str, "data...")
    try:
        weather_df = pd.read_excel(weather_df_str) 
        print("✅ Loading", weather_df_str, "data has been successfully completed.")
    except:
        print("⛔️ Error on loading", weather_df_str, "data!")
    return (df, weather_df, output_data, test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data preprocesser application, BBVDATA hackathon")

    parser.add_argument("-input", "-i", dest="in_df", help="Input dataframe filename (.csv)")
    parser.add_argument("-weather", "-w", dest="weather_df", help="Weather dataframe filename (.xlsx)")
    parser.add_argument("-output", "-o", dest="out_df", help="Final preprocessed filename")
    parser.add_argument("--test", dest="test", default=False, help="Preprocess for testing (default: False)", type=bool)

    args = parser.parse_args()
    data = load_data(args.in_df, args.weather_df, args.out_df, args.test)

    preprocess(*data)