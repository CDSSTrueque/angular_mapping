#!/usr/bin/env python

from pysolar.solar import *

from math import *
import numpy as np


class solarPos:
    def __init__(self, LAT, LON, date_, time_):
        #--- Args needed ---
        self.lat = LAT
        self.lon = LON

        self.year   = date_[0]
        self.month  = date_[1]
        self.day    = date_[2]

        self.hour   = time_[0]
        self.minute = time_[1]
        self.second = time_[2]
        self.microsecond = 0
        #---------------------

        #--- Optional args ---
        self.imgs          = []
        self.im_name       = []
        self.id_im         = 0
        self.semi_sphere   = False
        self.view_proj_vec = False
        self.pysolar_flag  = True

        d = datetime.datetime(self.year, self.month, self.day, self.hour, self.minute, self.second, 0)

        self.date_time= []
        self.altitude = []
        self.azimuth = []
        self.p_x = []
        self.p_y = []
        self.p_z = []
        self.point = []

        self.origin =   []
        self.origin_x = []
        self.origin_y = []
        self.origin_z = []

        self.sun_path = []

        self.p_x_img = []
        self.p_y_img = []

    #-- End Init --

    def compute_position(self):

        d = datetime.datetime(
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
            tzinfo=datetime.timezone(datetime.timedelta(hours=-7))
        )

        #print(d)
        alt = get_altitude(self.lat, self.lon, d)
        az = get_azimuth(self.lat, self.lon, d)
        self.altitude.append(alt)
        self.azimuth.append(az)
        self.p_x.append(cos(alt*pi/180)*cos(az*pi/180))
        self.p_y.append(cos(alt*pi/180)*sin(az*pi/180))
        self.p_z.append(sin(alt*pi/180))
        #print("azimuth")
        #print(self.azimuth)
        #print("-------------")
        #print("altitude")
        #print(self.altitude)
        #print("-------------")
        points = [self.p_x, self.p_y, self.p_z]
        return points
    #-- End Compute position --

    def get_azimuth_values(self):
        return self.azimuth

    def get_altitude_values(self):
        return self.altitude
