import googlemaps
from datetime import datetime, timedelta
from itertools import combinations
import pickle
import sys
import csv


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
RES = pickle.load(open(filename,"rb"))
# try:
#     RES = pickle.load(open(filename, "rb"))
# except NameError:
#
RES = []
try:
    compute_distance(st[0], st[1], departure_time=TIME_TEST[2])
except googlemaps.exceptions.Timeout:
    print('rer')


for n in cond:
    if n[3]==False:
        st=n[:2]
        t=n[2]
        try:
            departure_time = datetime.strptime('Aug 1 2017 ' + t, '%b %d %Y %I:%M%p')
            res = compute_distance(st[0], st[1], departure_time=departure_time)
            RES.append([res['place1'],res['place2'],res['departure_time'],res['duration_walking'],res['distance_walking'],res['duration_transit'],res['distance_transit'],departure_time,st[0],st[1]])
            status = 'OK'
        except googlemaps.exceptions.Timeout:
            print('reached quota')
            break
        except:
            RES.append([st[0], st[1], None, None, None, None, None, departure_time, st[0], st[1]])
            status = 'FAILED'
            print(sys.exc_info()[0])
        pickle.dump(RES, open(filename, "wb"))
        print( round(100 * 100 * len(RES)/all_cond_number)/100,st[0], st[1],status)

sched, lnd_st = [], []
with open('/Users/joeDiHare/Documents/TheRunnerBlogPost/data/underground-stations.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader: lnd_st.append(row)
with open('/Users/joeDiHare/Documents/TheRunnerBlogPost/data/underground-travel-time.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader: sched.append(row)