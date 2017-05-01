import googlemaps
from datetime import datetime, timedelta
from itertools import combinations
import pickle

api_key = 'AIzaSyCmFwfAh_UA2j2uQgGtKatCBDbCERdHzHk'
gmaps = googlemaps.Client(key=api_key)

cities=['LND','NPL','WDC']
suffix={'LND':' underground station, London, UK',
        'NPL':' metropolitana, Napoli',
        'WDC':' metro station, USA'}
stations=[]
for c in cities:
    tmp=[]
    with open("/Users/joeDiHare/Documents/MATLAB/"+c+"_train_dist.csv", "r") as filestream:
        for line in filestream:
            tmp = [_+suffix[c] for _ in line.split(",")]
    stations.append(tmp)
# # Geocoding an address
# geocode_result = gmaps.geocode('1600 Amphitheatre Parkway, Mountain View, CA')
# # Look up an address with reverse geocoding
# reverse_geocode_result = gmaps.reverse_geocode((40.714224, -73.961452))
# Request directions via public transit
# now = datetime.now()
# for mode in ["walking", "transit"]:
#     directions_result = gmaps.directions("Mansion House Station, 38 Cannon St, London EC4N 6JD, UK",
#                                      "Cannon Street Underground Station, Cannon St, London EC4N 6AP, UK",
#                                      mode=mode,#"walking",#"transit", #
#                                      transit_mode='rail',# bus, subway, train,
#                                      departure_time=now)
#     print(mode,directions_result[0]['legs'][0]['duration']['text'],directions_result[0]['legs'][0]['distance']['text'])

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


def compute_distance(place1, place2, departure_time=datetime.today()+timedelta(days=1), transit_mode='rail'):
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


# directions_result = gmaps.directions(place1, place2,mode='walking',transit_mode='rail',departure_time=datetime.today())
# res = compute_distance(stations[0][1],stations[0][2])
# test_combinations(stations[0][1],stations[0][2])

# place1=stations[1][1]
# for place2 in stations[1]:
#     if test_combinations(place1,place2):
#         print(place1+' and '+place2+' are candidates')
#     else:
#         print(place1 + ' and ' + place2 + ' NOT candidates')
#
# for t in ['06:00AM','09:00AM','05:00PM','08:00PM','10:00PM']:
#     r1=compute_distance(stations[0][14],stations[0][13], departure_time=datetime.strptime('Apr 24 2017 '+t, '%b %d %Y %I:%M%p'))
#     print(t,r1['duration_walking'],r1['duration_transit'])


# Create all combinations
all_comb=[]
for L in stations:
    all_comb.append([comb for comb in combinations(L, 2)])

TIME_TEST = ['06:00AM','09:00AM','05:00PM','08:00PM','10:00PM']
RES = []
for cities_station in all_comb:
    for st in cities_station:
        for t in TIME_TEST:
            try:
                departure_time = datetime.strptime('Apr 29 2017 ' + t, '%b %d %Y %I:%M%p')
                res = compute_distance(st[0], st[1], departure_time=departure_time)
                RES.append([res['place1'],res['place2'],res['departure_time'],res['duration_walking'],res['distance_walking'],res['duration_transit'],res['distance_transit'],departure_time,st[0],st[1]])
            except:
                RES.append([st[0], st[1], None, None, None, None, None, departure_time,st[0], st[1]])
            pickle.dump(RES, open("results_theRunner.p", "wb"))
        print(100*len(RES)/sum([len(x) for x in all_comb]))