a = '''
rotate=no
header=no
msgdat=1

symsiz=2 : labsiz=2 : annsiz=2
yhor=y

wylen=11
wxlen=11


; DEFINE TIC MARKS
ticdir=inward
ticsiz=1.3

xlen = 4 : ylen = 4

a=1.5 ;x offset
b=4.2 ;x scale
c=-1 ;y offset
d=3.2 ;y scale

w=0.2;symbol offset

clip=1

msgsiz=2
6,13.5"Unilateral
CAB (1000 pps)"

;                       >>>  Data Section  <<<
; ------------------------------------------------------------
newframe

xllc=a+0*b : yllc=c+1*d

xpercent=90 : xmin=22 : xmax=1 : xint=21
xanskp=2
xLabel=Left Electrode Number
xannot=20,20 16,16 12,12 8,8 4,4

ypercent=90 : ymin=0 : ymax=100 : yint=5.5
ylabel=Proportion Higher
yannot=

pltcol=0
pltype=line
lintype=5
pltlwt=2
data
20 -100
20 200
plot
12 -100
12 200
plot
4 -100
4 200
plot

0 50
25 50
plot
lintype=0
pltlwt=2

;L20
pltype=both
symbol=1
pltcol=1
xofst=0
17	100
18	96.66666667
19	96.66666667
20	43.33333333
21	6.666666667
22	0
plot

;L12
symbol=1
9	100
10	100
11	100
12	40
13	6.666666667
14	0
15	0
plot

;L4
symbol=1
1	100
2	90
3	76.66666667
4	56.66666667
5	16.66666667
6	6.666666667
7	0
plot

; ------------------------------------------------------------
newframe

xllc=a+1*b : yllc=c+1*d

xpercent=90 : xmin=22 : xmax=1 : xint=21
xanskp=2
xLabel=Right Electrode Number
xannot=20,20 16,16 12,12 8,8 4,4

ypercent=80 : ymin=0 : ymax=100 : yint=5.5
ylabel=
yanskp=-1

pltcol=0
pltype=line
lintype=5
pltlwt=2
data
20 -100
20 200
plot
12 -100
12 200
plot
4 -100
4 200
plot

0 50
25 50
plot
lintype=0
pltlwt=2

;R20
pltype=both
symbol=1
pltcol=4
xofst=0
17	83.33333333
18	93.33333333
19	66.66666667
20	50
21	16.66666667
22	10
plot

;R12
symbol=1
9	100
10	100
11	96.66666667
12	40
13	13.33333333
14	3.333333333
15	0
plot

;R4
symbol=1
1	83.33333333
2	63.33333333
3	70
4	60
5	40
6	10
7	10
plot
'''