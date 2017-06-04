import googlemaps
from datetime import datetime, timedelta
from itertools import combinations
import pickle
import sys
import csv
import pandas as pd
from pandas import DataFrame


api_key = 'AIzaSyCmFwfAh_UA2j2uQgGtKatCBDbCERdHzHk'
gmaps = googlemaps.Client(key=api_key)

cities=['NPL']#,'WDC','LND']
suffix={'LND':' underground station, London, UK',
        'NPL':' metropolitana, Napoli',
        'WDC':' metro station, USA'}
stations=[]
for c in cities:
    tmp=[]
    with open("/Users/stecose/Documents/TheRunnerBlogPost/"+c+"_train_dist_lines.csv", "r") as filestream:
        for line in filestream:
            tmp = line.split(",")
            tmp = [_+suffix[c] for _ in tmp[::2]]
    stations.append(tmp)

def test_combinations(place1, place2):
    try:
        TIME_TEST=['peak-time']
        TIME_TEST=[datetime.strptime('Apr 24 2017  1:30PM', '%b %d %Y %I:%M%p')]
        # datetime.today() + timedelta(days=1)
        for time in TIME_TEST:
            res = compute_distance(place1, place2,departure_time=time)
            if is_not_a_valid_candidate(res):
                return False
        return True
    except:
        print('some other error')
        return False

def compute_distance(place1, place2, departure_time=datetime.today()+timedelta(days=7), transit_mode='rail'):
    res = {}
    # for mode in ["walking", "transit"]:
    directions_result = gmaps.directions(place1, place2,mode='walking',transit_mode=transit_mode,departure_time=departure_time)
    res['duration_walking'] = directions_result[0]['legs'][0]['duration']['text']
    res['distance_walking'] = directions_result[0]['legs'][0]['distance']['text']
    directions_result = gmaps.directions(place1, place2, mode='transit', transit_mode=transit_mode,departure_time=departure_time)
    res['duration_transit'] = directions_result[0]['legs'][0]['duration']['text']
    res['distance_transit'] = directions_result[0]['legs'][0]['distance']['text']
    res['place1'] = place1
    res['place2'] = place2
    # res['transit_mode'] = transit_mode
    res['departure_time'] = departure_time
    return res

def is_not_a_valid_candidate(res):
#     check if distance is within X km and duration within Y min and ratio
    if int(res['duration_walking'].split()[0]) >= 30:
        print('too long:',res['duration_walking'])
        return True
    if float(res['distance_walking'].split()[0]) >= 1.5:
        print('too far:',res['distance_walking'])
        return True
    if int(res['duration_transit'].split()[0]) >= 30:
        print('too long:',res['duration_transit'])
        return True
    if float(res['distance_transit'].split()[0]) >= 1.5:
        print('too far:',res['distance_transit'])
        return True
    if int(res['duration_walking'].split()[0]) / int(res['duration_transit'].split()[0]) > 3:
        print('duration ratio too big')
        return True
    return False

# Create all combinations
all_comb=[]
for L in stations:
    all_comb.append([comb for comb in combinations(L, 2)])
all_cond_number = sum([len(x) for x in all_comb])

TIME_TEST = ['06:00AM','09:00AM','05:00PM','08:00PM','10:00PM']
cond=[]
for cities_station in all_comb:
    for st in cities_station:
        for t in TIME_TEST:
            cond.append([st[0], st[1], t, False])

filename="results_theRunner.p"
# RES = pickle.load(open(filename,"rb"))
# try:
#     RES = pickle.load(open(filename, "rb"))
# except NameError:
#
# RES = []
# for n in cond:
#     if n[3]==False:
#         st=n[:2]
#         t=n[2]
#         try:
#             departure_time = datetime.strptime('Aug 1 2017 ' + t, '%b %d %Y %I:%M%p')
#             res = compute_distance(st[0], st[1], departure_time=departure_time)
#             RES.append([res['place1'],res['place2'],res['departure_time'],res['duration_walking'],res['distance_walking'],res['duration_transit'],res['distance_transit'],departure_time,st[0],st[1]])
#             status = 'OK'
#         except googlemaps.exceptions.Timeout:
#             print('reached quota')
#             break
#         except:
#             RES.append([st[0], st[1], None, None, None, None, None, departure_time, st[0], st[1]])
#             status = 'FAILED'
#             print(sys.exc_info()[0])
#         # pickle.dump(RES, open(filename, "wb"))
#         print( round(100 * 100 * len(RES)/all_cond_number)/100,st[0], st[1],status)


######################################
# Get travel time for all combinations
######################################
def parse_time(time):
    # parse time (string) to integer value in minutes
    spl_time = time.split(' ')
    h, m = 0, 0
    for i, s in enumerate(spl_time):
        if 'hour' == s[:4]:
            h = int(spl_time[i - 1]) * 60
    for i, s in enumerate(spl_time):
        if 'min' == s[:3]:
            m = float(spl_time[i - 1])
    return h+m

sched, lnd_st = [], []
with open('/Users/joeDiHare/Documents/TheRunnerBlogPost/data/underground-stations.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader: lnd_st.append(row)
with open('/Users/joeDiHare/Documents/TheRunnerBlogPost/data/underground-travel-time.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader: sched.append(row)
 # create table of conditions
TIME_TEST = ['08:00AM','05:00PM','08:00PM']
RES, cond =[], []
for n in sched[1::]:
    for t in TIME_TEST:
        cond.append([lnd_st[int(n[0])][3], lnd_st[int(n[1])][3], t]+ n)
for n in cond[432::]:
    departure_time = datetime.strptime('Jun 7 2017 ' + n[2], '%b %d %Y %I:%M%p')
    res = compute_distance(n[0]+' underground station, London, UK', n[1]+' underground station, London, UK', departure_time=departure_time)
    RES.append([n[0],n[1],
                res['departure_time'],res['duration_walking'],res['distance_walking'],res['duration_transit'],res['distance_transit'],
                parse_time(res['duration_walking'])/parse_time(res['duration_transit'])])

# n=['Mansion House','Cannon Street']
# for t in TIME_TEST:
#     departure_time = datetime.strptime('Jun 7 2017 ' + t, '%b %d %Y %I:%M%p')
#     res = compute_distance(n[0] + ' underground station, London, UK', n[1] + ' underground station, London, UK',
#                            departure_time=departure_time)
#     print([n[0],n[1],res['departure_time'],res['duration_walking'],res['distance_walking'],res['duration_transit'],res['distance_transit']])



sel, R=[], []
for n in RES:
    R.append(n + [parse_time(n[3])/parse_time(n[5])])
    if R[-1][-1]<=2.5:
        sel.append(R[-1])
        print(sel[-1])

headers=['from','to','dep_time','dur_walking','dis_walking','dur_transit','dist_transit','ratio']
df = DataFrame(RES, columns=headers)

