from influxdb_client import InfluxDBClient

URL = "https://data.sailgp.tech"
ORG = "0c2a130d50b8facc"
BUCKET = "sailgp"
TOKEN = "2vTlG__z6bc7bibptc1FE_gXRwK6761dmxW_sasiAC1qsNqwbAbAj0PJD9yRIQPR0bfwdl_4-S_5gIecgkfz_Q=="

client = InfluxDBClient(url=URL, token=TOKEN, org=ORG, verify_ssl=False)
query_api = client.query_api()

try:
    query = f'import "influxdata/influxdb/schema" schema.measurements(bucket: "{BUCKET}")'
    result = query_api.query_data_frame(org=ORG, query=query)
    print(result.head())
    print("✅ Connexion InfluxDB OK")
except Exception as e:
    print("❌ Erreur :", e)
finally:
    client.close()
