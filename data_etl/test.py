import pandas as pd
import glob
from dotenv import dotenv_values
import requests
import json
from datetime import datetime
import os


geo_json = {
    "강원": {
        "강릉": [
            37.750894,
            128.878083
        ],
        "고성": [
            34.972993450000004,
            128.32222422117505
        ],
        "동해": [
            37.524763,
            129.114492
        ],
        "삼척": [
            37.4498902,
            129.16518225
        ],
        "속초": [
            38.2069014,
            128.591768
        ],
        "양구": [
            38.1099784,
            127.98994971680318
        ],
        "양양": [
            38.075494,
            128.619005
        ],
        "영월": [
            37.1836962,
            128.4618391
        ],
        "원주": [
            37.340064,
            127.920172
        ],
        "인제": [
            38.06972245,
            128.1703896658264
        ],
        "정선": [
            37.3806106,
            128.66085599814173
        ],
        "철원": [
            38.14676385,
            127.31339020664066
        ],
        "춘천": [
            37.8813828,
            127.73020409160627
        ],
        "태백": [
            37.1640972,
            128.9857646127099
        ],
        "평창": [
            37.3707848,
            128.390305
        ],
        "홍천": [
            37.6972227,
            127.8887607
        ],
        "화천": [
            38.1062163,
            127.70822869212424
        ],
        "횡성": [
            37.4917396,
            127.9851205
        ]
    },
    "경기": {
        "가평": [
            37.8323555,
            127.5121384
        ],
        "고양": [
            37.6560818,
            126.8318519
        ],
        "과천": [
            37.4292215,
            126.98759439966764
        ],
        "광명": [
            37.47857005,
            126.864662543934
        ],
        "구리": [
            37.594351200000006,
            127.12964717040356
        ],
        "군포": [
            37.3614384,
            126.93537331130457
        ],
        "김포": [
            37.615489350000004,
            126.71542848762226
        ],
        "남양주": [
            37.6353786,
            127.2171458
        ],
        "동두천": [
            37.9035691,
            127.0604829
        ],
        "부천": [
            37.504678,
            126.76399664105361
        ],
        "성남": [
            37.420094,
            127.1266095
        ],
        "수원": [
            37.26165,
            127.031849
        ],
        "시흥": [
            37.382025,
            126.8058162
        ],
        "안산": [
            37.32202065,
            126.83083395023425
        ],
        "안성": [
            37.0080982,
            127.2797684197808
        ],
        "안양": [
            37.39428675,
            126.95686815
        ],
        "양주": [
            37.7853275,
            127.04579779667935
        ],
        "양평": [
            37.49166665,
            127.4875344286406
        ],
        "여주": [
            37.29822935,
            127.63714056925167
        ],
        "연천": [
            38.0965205,
            127.07528053825452
        ],
        "오산": [
            37.1498559,
            127.077485
        ],
        "용인": [
            37.24106295,
            127.17763276355933
        ],
        "의왕": [
            37.3444099,
            126.9703907
        ],
        "의정부": [
            37.73929855,
            127.03487339253779
        ],
        "이천": [
            37.27234835,
            127.43501673062367
        ],
        "파주": [
            37.7587244,
            126.7783565
        ],
        "평택": [
            36.9914004,
            127.11300249881845
        ],
        "포천": [
            37.89493075,
            127.20035206746208
        ],
        "하남": [
            37.5417742,
            127.2067037
        ],
        "화성": [
            37.1990931,
            126.83096849576879
        ],
        "광주": [
            35.160104849999996,
            126.85146886317503
        ]
    },
    "경상남도": {
        "거제": [
            34.880481,
            128.6212633
        ],
        "거창": [
            35.6860981,
            127.9096955
        ],
        "고성": [
            34.972993450000004,
            128.32222422117505
        ],
        "김해": [
            35.2270823,
            128.8903754
        ],
        "사천": [
            35.0032087,
            128.06464808082376
        ],
        "양산": [
            35.3349557,
            129.0355279
        ],
        "의령": [
            35.3221349,
            128.26150775398003
        ],
        "진주": [
            35.1794262,
            128.10765
        ],
        "창녕": [
            35.544625350000004,
            128.49218204844112
        ],
        "창원": [
            35.2278577,
            128.6818148
        ],
        "통영": [
            34.853924,
            128.434288
        ],
        "하동": [
            35.0673125,
            127.75131316642386
        ],
        "함안": [
            35.2724799,
            128.40650842122648
        ],
        "함양": [
            35.5205424,
            127.7251840806793
        ],
        "합천": [
            35.566666,
            128.1658222
        ],
        "남해": [
            34.8374773,
            127.8923272049895
        ],
        "밀양": [
            35.5036457,
            128.7460822065127
        ],
        "산청": [
            35.4155607,
            127.87347274953983
        ]
    },
    "경상북도": {
        "경산": [
            35.8250934,
            128.741514
        ],
        "경주": [
            35.85607365,
            129.22544639039333
        ],
        "고령": [
            35.7260865,
            128.26289467152537
        ],
        "군위": [
            36.24303235,
            128.57301266158566
        ],
        "김천": [
            36.13988245,
            128.1136467524054
        ],
        "문경": [
            36.5865788,
            128.1871654429272
        ],
        "상주": [
            36.411013,
            128.15910808096302
        ],
        "성주": [
            35.919224,
            128.28307226053704
        ],
        "안동": [
            36.56841945,
            128.72960779019223
        ],
        "영덕": [
            36.4150229,
            129.3652650993373
        ],
        "영양": [
            36.6667736,
            129.11249780499207
        ],
        "영주": [
            36.80567425,
            128.6240640597999
        ],
        "영천": [
            35.973243749999995,
            128.9386464288963
        ],
        "예천": [
            36.646849700000004,
            128.43720577246168
        ],
        "울릉": [
            37.48442455,
            130.9057808
        ],
        "울진": [
            36.993076849999994,
            129.40063449657123
        ],
        "의성": [
            36.35270335,
            128.69717600985848
        ],
        "청도": [
            35.647432,
            128.734327
        ],
        "청송": [
            36.43627225,
            129.05709287719324
        ],
        "칠곡": [
            35.9955834,
            128.40176349844356
        ],
        "포항": [
            36.0190154,
            129.3432304248518
        ],
        "구미": [
            36.1196537,
            128.3442731316831
        ],
        "봉화": [
            36.893128399999995,
            128.73249203428415
        ]
    },
    "경남": {
        "거제": [
            34.880481,
            128.6212633
        ],
        "거창": [
            35.6860981,
            127.9096955
        ],
        "고성": [
            34.972993450000004,
            128.32222422117505
        ],
        "김해": [
            35.2270823,
            128.8903754
        ],
        "사천": [
            35.0032087,
            128.06464808082376
        ],
        "양산": [
            35.3349557,
            129.0355279
        ],
        "의령": [
            35.3221349,
            128.26150775398003
        ],
        "진주": [
            35.1794262,
            128.10765
        ],
        "창녕": [
            35.544625350000004,
            128.49218204844112
        ],
        "창원": [
            35.2278577,
            128.6818148
        ],
        "통영": [
            34.853924,
            128.434288
        ],
        "하동": [
            35.0673125,
            127.75131316642386
        ],
        "함안": [
            35.2724799,
            128.40650842122648
        ],
        "함양": [
            35.5205424,
            127.7251840806793
        ],
        "합천": [
            35.566666,
            128.1658222
        ],
        "남해": [
            34.8374773,
            127.8923272049895
        ],
        "밀양": [
            35.5036457,
            128.7460822065127
        ],
        "산청": [
            35.4155607,
            127.87347274953983
        ]
    },
    "경북": {
        "경산": [
            35.8250934,
            128.741514
        ],
        "경주": [
            35.85607365,
            129.22544639039333
        ],
        "고령": [
            35.7260865,
            128.26289467152537
        ],
        "군위": [
            36.24303235,
            128.57301266158566
        ],
        "김천": [
            36.13988245,
            128.1136467524054
        ],
        "문경": [
            36.5865788,
            128.1871654429272
        ],
        "상주": [
            36.411013,
            128.15910808096302
        ],
        "성주": [
            35.919224,
            128.28307226053704
        ],
        "안동": [
            36.56841945,
            128.72960779019223
        ],
        "영덕": [
            36.4150229,
            129.3652650993373
        ],
        "영양": [
            36.6667736,
            129.11249780499207
        ],
        "영주": [
            36.80567425,
            128.6240640597999
        ],
        "영천": [
            35.973243749999995,
            128.9386464288963
        ],
        "예천": [
            36.646849700000004,
            128.43720577246168
        ],
        "울릉": [
            37.48442455,
            130.9057808
        ],
        "울진": [
            36.993076849999994,
            129.40063449657123
        ],
        "의성": [
            36.35270335,
            128.69717600985848
        ],
        "청도": [
            35.647432,
            128.734327
        ],
        "청송": [
            36.43627225,
            129.05709287719324
        ],
        "칠곡": [
            35.9955834,
            128.40176349844356
        ],
        "포항": [
            36.0190154,
            129.3432304248518
        ],
        "구미": [
            36.1196537,
            128.3442731316831
        ],
        "봉화": [
            36.893128399999995,
            128.73249203428415
        ]
    },
    "광주광역시": {
        "광주광역": [
            35.160104849999996,
            126.85146886317503
        ]
    },
    "대구광역시": {
        "달성": [
            35.7748111,
            128.43157371455408
        ],
        "대구광역": [
            35.8715082,
            128.6019174
        ]
    },
    "대전": {
        "대전": [
            36.3505508,
            127.38496083451648
        ]
    },
    "부산": {
        "기장": [
            35.2443935,
            129.2221725426327
        ],
        "부산": [
            35.1799421,
            129.0752053597061
        ]
    },
    "서울": {
        "서울": [
            37.56678925,
            126.97842039866163
        ]
    },
    "세종": {
        "세종": [
            36.4801176,
            127.2891886504768
        ]
    },
    "울산": {
        "울주": [
            35.5217893,
            129.2432278
        ],
        "울산": [
            35.5395077,
            129.3112519
        ]
    },
    "인천광역시": {
        "옹진": [
            37.446617,
            126.63684979960178
        ],
        "강화": [
            37.74631185,
            126.4879634862095
        ],
        "인천광역": [
            37.4561224,
            126.70524077779223
        ]
    },
    "전라남도": {
        "강진": [
            34.64197225,
            126.76701057943214
        ],
        "곡성": [
            35.282024500000006,
            127.29197950715195
        ],
        "광양": [
            34.94042355,
            127.69595059082727
        ],
        "구례": [
            35.2022669,
            127.46322682459893
        ],
        "나주": [
            35.016066,
            126.7108372
        ],
        "담양": [
            35.3211872,
            126.98824159507429
        ],
        "목포": [
            34.81152025,
            126.39179345566497
        ],
        "무안": [
            34.990404,
            126.480564
        ],
        "보성": [
            34.77127955,
            127.07958280025423
        ],
        "순천": [
            34.9506588,
            127.4872797
        ],
        "신안": [
            34.83292435,
            126.3516619564478
        ],
        "여수": [
            34.7622079,
            127.6630984
        ],
        "영광": [
            35.27728315,
            126.51204493865788
        ],
        "영암": [
            34.8002341,
            126.69682538044813
        ],
        "완도": [
            34.31100655,
            126.75498725
        ],
        "장성": [
            35.3018515,
            126.78498276212852
        ],
        "장흥": [
            34.68164455,
            126.90709823414205
        ],
        "진도": [
            34.48683095,
            126.26352624298482
        ],
        "함평": [
            35.06594495,
            126.51663456029556
        ],
        "해남": [
            34.5740258,
            126.59925158096539
        ],
        "화순": [
            35.0645741,
            126.98645234028609
        ],
        "고흥": [
            34.6045193,
            127.27551852504146
        ]
    },
    "전라북도": {
        "김제": [
            35.8045264,
            126.8795519
        ],
        "남원": [
            35.41641765,
            127.39046103761552
        ],
        "무주": [
            36.0068424,
            127.6607719
        ],
        "부안": [
            35.73183405,
            126.7332666389581
        ],
        "순창": [
            35.374466049999995,
            127.13735262902176
        ],
        "완주": [
            35.9045703,
            127.16156605817667
        ],
        "익산": [
            35.9484017,
            126.95744654286905
        ],
        "임실": [
            35.61768885,
            127.28891168477702
        ],
        "장수": [
            35.6472636,
            127.52105476172001
        ],
        "전주": [
            35.824146150000004,
            127.14810962694804
        ],
        "정읍": [
            35.569921,
            126.85601058639435
        ],
        "진안": [
            35.791663,
            127.424812
        ],
        "고창": [
            35.4358084,
            126.7020971
        ],
        "군산": [
            35.9676041,
            126.73688162152297
        ]
    },
    "전남": {
        "강진": [
            34.64197225,
            126.76701057943214
        ],
        "곡성": [
            35.282024500000006,
            127.29197950715195
        ],
        "광양": [
            34.94042355,
            127.69595059082727
        ],
        "구례": [
            35.2022669,
            127.46322682459893
        ],
        "나주": [
            35.016066,
            126.7108372
        ],
        "담양": [
            35.3211872,
            126.98824159507429
        ],
        "목포": [
            34.81152025,
            126.39179345566497
        ],
        "무안": [
            34.990404,
            126.480564
        ],
        "보성": [
            34.77127955,
            127.07958280025423
        ],
        "순천": [
            34.9506588,
            127.4872797
        ],
        "신안": [
            34.83292435,
            126.3516619564478
        ],
        "여수": [
            34.7622079,
            127.6630984
        ],
        "영광": [
            35.27728315,
            126.51204493865788
        ],
        "영암": [
            34.8002341,
            126.69682538044813
        ],
        "완도": [
            34.31100655,
            126.75498725
        ],
        "장성": [
            35.3018515,
            126.78498276212852
        ],
        "장흥": [
            34.68164455,
            126.90709823414205
        ],
        "진도": [
            34.48683095,
            126.26352624298482
        ],
        "함평": [
            35.06594495,
            126.51663456029556
        ],
        "해남": [
            34.5740258,
            126.59925158096539
        ],
        "화순": [
            35.0645741,
            126.98645234028609
        ],
        "고흥": [
            34.6045193,
            127.27551852504146
        ]
    },
    "전북": {
        "김제": [
            35.8045264,
            126.8795519
        ],
        "남원": [
            35.41641765,
            127.39046103761552
        ],
        "무주": [
            36.0068424,
            127.6607719
        ],
        "부안": [
            35.73183405,
            126.7332666389581
        ],
        "순창": [
            35.374466049999995,
            127.13735262902176
        ],
        "완주": [
            35.9045703,
            127.16156605817667
        ],
        "익산": [
            35.9484017,
            126.95744654286905
        ],
        "임실": [
            35.61768885,
            127.28891168477702
        ],
        "장수": [
            35.6472636,
            127.52105476172001
        ],
        "전주": [
            35.824146150000004,
            127.14810962694804
        ],
        "정읍": [
            35.569921,
            126.85601058639435
        ],
        "진안": [
            35.791663,
            127.424812
        ],
        "고창": [
            35.4358084,
            126.7020971
        ],
        "군산": [
            35.9676041,
            126.73688162152297
        ]
    },
    "제주": {
        "서귀포": [
            33.25560095,
            126.51044279263246
        ],
        "제주": [
            33.4989146,
            126.5301288
        ]
    },
    "충청남도": {
        "계룡": [
            36.2744671,
            127.24851912218475
        ],
        "공주": [
            36.446787549999996,
            127.11911093564214
        ],
        "논산": [
            36.18723525,
            127.09874445090604
        ],
        "당진": [
            36.8899526,
            126.6457659
        ],
        "보령": [
            36.3362406,
            126.6503527
        ],
        "부여": [
            36.275276399999996,
            126.91005618985639
        ],
        "서산": [
            36.78490975,
            126.45036399498237
        ],
        "서천": [
            36.0781533,
            126.70290265612273
        ],
        "아산": [
            36.788298,
            127.0019212
        ],
        "예산": [
            36.6808741,
            126.8450906
        ],
        "천안": [
            36.815131300000004,
            127.11403824650358
        ],
        "청양": [
            36.45934525,
            126.80198276009386
        ],
        "태안": [
            36.745690100000004,
            126.29804624840764
        ],
        "홍성": [
            36.601289449999996,
            126.66080202437678
        ],
        "금산": [
            36.1088987,
            127.48805170036097
        ]
    },
    "충청북도": {
        "괴산": [
            36.8153353,
            127.78665065029344
        ],
        "단양": [
            36.9846702,
            128.36555363079424
        ],
        "영동": [
            36.175022999999996,
            127.78343145036558
        ],
        "옥천": [
            36.306508,
            127.5714798
        ],
        "음성": [
            36.94020325,
            127.69048095607026
        ],
        "제천": [
            37.1326615,
            128.191219
        ],
        "증평": [
            36.785431700000004,
            127.58161799060284
        ],
        "진천": [
            36.8554267,
            127.4357747
        ],
        "청주": [
            36.642582250000004,
            127.48945954958177
        ],
        "충주": [
            36.99102705,
            127.92602422754781
        ],
        "보은": [
            36.48927735,
            127.73036686080755
        ]
    },
    "충남": {
        "계룡": [
            36.2744671,
            127.24851912218475
        ],
        "공주": [
            36.446787549999996,
            127.11911093564214
        ],
        "논산": [
            36.18723525,
            127.09874445090604
        ],
        "당진": [
            36.8899526,
            126.6457659
        ],
        "보령": [
            "nodata",
            "nodata"
        ],
        "부여": [
            36.275276399999996,
            126.91005618985639
        ],
        "서산": [
            36.78490975,
            126.45036399498237
        ],
        "서천": [
            36.0781533,
            126.70290265612273
        ],
        "아산": [
            36.788298,
            127.0019212
        ],
        "예산": [
            36.6808741,
            126.8450906
        ],
        "천안": [
            36.815131300000004,
            127.11403824650358
        ],
        "청양": [
            36.45934525,
            126.80198276009386
        ],
        "태안": [
            36.745690100000004,
            126.29804624840764
        ],
        "홍성": [
            36.601289449999996,
            126.66080202437678
        ],
        "금산": [
            36.1088987,
            127.48805170036097
        ]
    },
    "충북": {
        "괴산": [
            36.8153353,
            127.78665065029344
        ],
        "단양": [
            36.9846702,
            128.36555363079424
        ],
        "영동": [
            36.175022999999996,
            127.78343145036558
        ],
        "옥천": [
            36.306508,
            127.5714798
        ],
        "음성": [
            36.94020325,
            127.69048095607026
        ],
        "제천": [
            37.1326615,
            128.191219
        ],
        "증평": [
            36.785431700000004,
            127.58161799060284
        ],
        "진천": [
            36.8554267,
            127.4357747
        ],
        "청주": [
            36.642582250000004,
            127.48945954958177
        ],
        "충주": [
            36.99102705,
            127.92602422754781
        ],
        "보은": [
            36.48927735,
            127.73036686080755
        ]
    }
}



def search_dic(data, data_dic):
    dic_keys = data_dic.keys()
    result = [None, None]
    complete = False
    print(f'data:{data}')
    for key in dic_keys:
        print(f'key:{key}')
        if data.find(key) >=0:
            print(f'통과')
            # print(f'data_dic[key]:{data_dic[key]}]')
            secon_keys = data_dic[key].keys()
            for skey in secon_keys:
                if data.find(skey) >=0:
                    result = [key, skey]
                    complete = True
                    break
        if complete:
            break
    
    return result

def create_geo_info(other_item, geo_json):
    result = []
    for df_item in other_item:
        table = df_item[0]
        df = df_item[1]
        # reference_col = json_dic[table]['reference_col']
        reference_col = 'FARM_LOCPLC'
        df['위도'] = None
        df['경도'] = None
        # print(f'table: {table}')
        # print(f'reference_col: {reference_col}')
        df.dropna(subset=[reference_col], inplace=True)
        df.reset_index(drop=True, inplace=True)

        for i in range(len(df)):
            ref_data = df.loc[i, reference_col]
            # print(f'i: {i}')
            # print(f'df_i: {df.iloc[i]}')
            print(f'ref_data: {ref_data}')
            print(f'reference_col: {reference_col}')
            (key, skey) = search_dic(ref_data, geo_json)
            if key != None:
                print(f'key: {key}')
                print(f'skey: {skey}')
                df.loc[i,'위도'] = geo_json[key][skey][0]
                df.loc[i,'경도'] = geo_json[key][skey][1]
        
        print(df)
        result.append([table, df])
    
    return result


cdf = pd.DataFrame(
    data = [   
        [1,"00000230","돼지오제스키병","정지창","4146125029","경기도 용인시 처인구 포곡읍 신원리","20030530","413000","돼지","1","6410601","경기 남부지소","20040403"],
        [2,"00000232","결핵병","수연목장","4278025328","강원도 철원군 김화읍 청양리","20030530","412004","소-젖소","26","6420086","강원 가축위생시험소",""],
        [3,"00000233","뉴캣슬병","박도균","4617043034","전라남도 나주시 봉황면 덕림리","20030324","415006","닭-육계","32200","6460117","전남 축산기술연구소",""],
        [4,"00000234","가금티프스","오세남","4415037022","충청남도 공주시 정안면 상룡리","20030318","415006","닭-육계","6500","6440451","충남 공주지소",""],
        [5,"00000235","가금티프스","김종엽","4159037021","경기도 화성시 장안면 어은리","20030410","415006","닭-육계","1200","6410598","경기도축산위생연구소",""],
        [5,"00000235","가금티프스","김종엽","4159037021","경상남도 고성군 고성읍 무량리","20030410","415006","닭-육계","1200","6410598","경기도축산위생연구소",""],
        [5,"00000235","가금티프스","김종엽","4159037021","경상북도 영천시 금호읍 구암리","20030410","415006","닭-육계","1200","6410598","경기도축산위생연구소",""],
        [5,"00000235","가금티프스","김종엽","4159037021","전라남도 함평군 신광면 동정리","20030410","415006","닭-육계","1200","6410598","경기도축산위생연구소",""],
        [5,"00000235","가금티프스","김종엽","4159037021","경상남도 창원시 마산합포구 진동면 교동리","20030410","415006","닭-육계","1200","6410598","경기도축산위생연구소",""],
    ],
    columns= ["ROW_NUM","ICTSD_OCCRRNC_NO","LKNTS_NM","FARM_NM","FARM_LOCPLC_LEGALDONG_CODE","FARM_LOCPLC","OCCRRNC_DE","LVSTCKSPC_CODE","LVSTCKSPC_NM","OCCRRNC_LVSTCKCNT","DGNSS_ENGN_CODE","DGNSS_ENGN_NM","CESSATION_DE"]
)
other_item = [['cattle',cdf],]

result = create_geo_info(other_item, geo_json)
print("result_df:\n", result[0][1].loc[:,['FARM_LOCPLC','위도','경도']].head())

# (key, skey) = search_dic("제주특별자치도 제주시 한림읍 상명리", geo_json)
# print("key:", key)
# print("skey:", skey)