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

# cities=['NPL']#,'WDC','LND']
# suffix={'LND':' underground station, London, UK',
#         'NPL':' metropolitana, Napoli',
#         'WDC':' metro station, USA'}
# stations=[]
# for c in cities:
#     tmp=[]
#     with open("/Users/stecose/Documents/TheRunnerBlogPost/"+c+"_train_dist_lines.csv", "r") as filestream:
#         for line in filestream:
#             tmp = line.split(",")
#             tmp = [_+suffix[c] for _ in tmp[::2]]
#     stations.append(tmp)


def compute_distance(place1, place2, departure_time=datetime.today()+timedelta(days=7), transit_mode='subway'):
    res = {}
    res['duration_walking'] = 'NA'
    res['distance_walking'] = 'NA'
    res['duration_transit'] = 'NA'
    res['distance_transit'] = 'NA'
    # for mode in ["walking", "transit"]:
    directions_result = gmaps.directions(place1, place2,mode='walking',departure_time=departure_time)
    if directions_result:
        res['duration_walking'] = directions_result[0]['legs'][0]['duration']['text']
        res['distance_walking'] = directions_result[0]['legs'][0]['distance']['text']
    directions_result = gmaps.directions(place1, place2, mode='transit', transit_mode=transit_mode,departure_time=departure_time)
    if directions_result:
        res['duration_transit'] = directions_result[0]['legs'][0]['duration']['text']
        res['distance_transit'] = directions_result[0]['legs'][0]['distance']['text']
    res['place1'] = place1
    res['place2'] = place2
    res['departure_time'] = departure_time
    return res

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
    return h + m


######################################
# Get travel time for all combinations
######################################
filename="results_theRunner.p"


lines, sched, lnd_st = [], [], []
with open('/Users/joeDiHare/Documents/TheRunnerBlogPost/data/underground-stations.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader: lnd_st.append(row)
with open('/Users/joeDiHare/Documents/TheRunnerBlogPost/data/underground-routes.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader: lines.append(row)
with open('/Users/joeDiHare/Documents/TheRunnerBlogPost/data/underground-travel-time.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader: sched.append(row)

# create table of conditions
TIME_TEST = ['08:00AM','02:00PM','05:00PM']
cond =[]
for n in sched[1::]:
    for t in TIME_TEST:
        cond.append([str(lnd_st[int(n[0])][1]) + ',' + str(lnd_st[int(n[0])][2]),
                     str(lnd_st[int(n[1])][1]) + ',' + str(lnd_st[int(n[1])][2]),
                     lnd_st[int(n[0])][3], lnd_st[int(n[1])][3], t]+ [int(_) for _ in n])
RES = []
for n in cond:
    departure_time = datetime.strptime('Jun 7 2017 ' + n[4], '%b %d %Y %I:%M%p')
    res = compute_distance(n[2] + ' underground station, London, UK',
                           n[3] + ' underground station, London, UK', departure_time=departure_time)
    ratio = parse_time(res['duration_walking'])/parse_time(res['duration_transit']) if parse_time(res['duration_transit'])!=0 else 'NA'
    RES.append([n[2],n[3],
                res['departure_time'],res['duration_walking'],res['distance_walking'],res['duration_transit'],res['distance_transit'],
                ratio, lines[n[-2]][1], n[0],n[1],n[-1]])
pickle.dump(RES, open(filename, "wb"))

headers=['from','to','dep_time','dur_walking','dis_walking','dur_transit','dist_transit','ratio','line','geo-from','geo-to','time_old']
df = DataFrame(RES, columns=headers)

print(df.dur_transit,df.time_old)


import colorsys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from bokeh.plotting import figure, show
from bokeh.resources import CDN
from bokeh.io import output_notebook
output_notebook( resources=CDN )
lines       = pd.read_csv('/Users/joeDiHare/Documents/TheRunnerBlogPost/data/underground-routes.csv', index_col=0)
stations    = pd.read_csv('/Users/joeDiHare/Documents/TheRunnerBlogPost/data/underground-stations.csv', index_col=0)
connections = pd.read_csv('/Users/joeDiHare/Documents/TheRunnerBlogPost/data/underground-travel-time.csv')

graph = nx.Graph()

for connection_id, connection in connections.iterrows():
    station1_name = stations.loc[connection['station1']]['name']
    station2_name = stations.loc[connection['station2']]['name']
    graph.add_edge(station1_name, station2_name, time=connection['time'])

# add the connection between Bank and Monument manually
graph.add_edge('Bank', 'Monument', time=1)


def pseudocolor(val):
    h = (1.0 - val) * 120 / 360
    r, g, b = colorsys.hsv_to_rgb(h, 1., 1.)
    return r * 255, g * 255, b * 255

normed = stations[['longitude', 'latitude']]
normed = normed - normed.min()
normed = normed / normed.max()
locations = dict(zip(stations['name'], normed[['longitude', 'latitude']].values))
# pageranks = dict(zip(stations['name'], normed['pagerank'].values))

p = figure(
    x_range=(.4, .7),
    y_range=(.2, .5),
    height=700,
    width=900,
)
for edge in graph.edges():
    p.line(
        x=[locations[pt][0] for pt in edge],
        y=[locations[pt][1] for pt in edge],
    )

for node in graph.nodes():
    x = [locations[node][0]]
    y = [locations[node][1]]
    p.circle(
        x, y,
        radius=.3,
        fill_color=pseudocolor(1),
        line_alpha=0)
    p.text(
        x, y,
        text={'value': node},
        # text_font_size=str(min(pageranks[node] * 12, 10)) + "pt",
        # text_alpha=pageranks[node],
        text_align='center',
        text_font_style='bold')

show(p)


# n=['Mansion House','Cannon Street']
# n=['Baker Street','Marylebone']
# n=['Baker Street Station, Marylebone Rd, Marylebone, London NW1, UK','Marylebone Railway Station, Melcombe Pl, London NW1 6JJ, UK']
# for t in ['08:00PM']:#,'08:02PM','08:04PM','08:06PM']:
#     departure_time = datetime.strptime('Jun 7 2017 ' + t, '%b %d %Y %I:%M%p')
#     res = compute_distance(n[0] + ' underground station, London, UK', n[1] + ' underground station, London, UK', departure_time=departure_time)
#     print([n[0],n[1],res['departure_time'],res['duration_walking'],res['distance_walking'],res['duration_transit'],res['distance_transit']])
# directions_result = gmaps.directions(n[0]+ ' underground station, London, UK', n[1]+ ' underground station, London, UK',mode='transit', transit_mode='subway',departure_time=departure_time)




# def test_combinations(place1, place2):
#     try:
#         TIME_TEST=[datetime.strptime('Apr 24 2017  1:30PM', '%b %d %Y %I:%M%p')]
#         # datetime.today() + timedelta(days=1)
#         for time in TIME_TEST:
#             res = compute_distance(place1, place2,departure_time=time)
#             if is_not_a_valid_candidate(res):
#                 return False
#         return True
#     except:
#         print('some other error')
#         return False
# def is_not_a_valid_candidate(res):
# #     check if distance is within X km and duration within Y min and ratio
#     if int(res['duration_walking'].split()[0]) >= 30:
#         print('too long:',res['duration_walking'])
#         return True
#     if float(res['distance_walking'].split()[0]) >= 1.5:
#         print('too far:',res['distance_walking'])
#         return True
#     if int(res['duration_transit'].split()[0]) >= 30:
#         print('too long:',res['duration_transit'])
#         return True
#     if float(res['distance_transit'].split()[0]) >= 1.5:
#         print('too far:',res['distance_transit'])
#         return True
#     if int(res['duration_walking'].split()[0]) / int(res['duration_transit'].split()[0]) > 3:
#         print('duration ratio too big')
#         return True
#     return False

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