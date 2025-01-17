from geopy.geocoders import Nominatim
import json
from dotenv import dotenv_values
import time

config = dotenv_values('.env')
LOCATIONS_JSON = config['LOCATIONS_JSON']
GIO_JSON = config['GIO_JSON']
location_json = {}
with open(LOCATIONS_JSON, 'r', encoding='utf-8') as f:
    location_json = json.load(f)

# Geolocator 초기화
geolocator = Nominatim(user_agent="South Korea")

locj_keys = location_json.keys()
result = {}
# 좌표 찾기
for lkj in locj_keys:
    result[lkj] = {}
    locations = location_json[lkj]
    loc_keys = locations.keys()
    for lk in loc_keys:
        location = locations[lk]

        loc = geolocator.geocode(location)
        if loc:
            print(f"{location}: {loc.latitude}, {loc.longitude}")
            result[lkj][lk] = [loc.latitude, loc.longitude]
        else:
            print(f"{location}의 좌표를 찾을 수 없습니다.")
            result[lkj][lk] = ['nodata', 'nodata']
        time.sleep(2)
    
with open(GIO_JSON, 'w', encoding="utf-8") as f:
    print("\nJSON 파일로 내보내는 중입니다.")
    json.dump(result, f, indent=4, ensure_ascii=False)
    print("파일 생성 완료")

