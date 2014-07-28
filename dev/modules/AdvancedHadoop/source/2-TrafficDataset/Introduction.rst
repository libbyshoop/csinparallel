***************************
Introduction to the Dataset
***************************

The Data
########

The UK department of Transportation keeps detailed records
of all traffic incidents. Fortunately for us they made this
data available to the public in the form of three csv files
that contain information about the accidents, casualties, 
and vehicles involved. 

These files are located on wmr in the ``/shared/bigData/traffic``
folder and are named Accidents7409.csv Casualty7409.csv and 
Vehicles7409.csv respectively.

Working with the Data
#####################

Each line in the files contains several fields seperated by
commas, to accesss these values, it is necessary to call
``key.split(',')`` (or the equivalent in whatever language
you're using) to get an array of values. If you want, you
can turn these values into an object, however it's faster
to simply refer to them by their index 

+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
| index | Accidents7409.csv                       | Casualty7409.csv                        | Vehicles7409.csv                 |
+=======+=========================================+=========================================+==================================+
|   0   | Accident Index                          | Accident Index                          | Accident Index                   |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   1   | Location Easting OSGR                   | Vehicle Reference                       | Vehicle Reference                |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   2   | Location Northing OSGR                  | Casualty Reference                      | Vehicle Type                     |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   3   | Longitude                               | Casualty Class                          | Towing/Articulation              |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   4   | Latitude                                | Sex of Casualty                         | Vehicle Maneuver                 |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   5   | Police Force                            | Age Band of Casualty                    | Vehicle Location Restricted Lane |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   6   | Accident Severity                       | Casualty Severity                       | Junction Location                |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   7   | Number of Vehicles                      | Pedestrian Location                     | Skidding/Overturning             |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   8   | Number of Casualties                    | Pedestrian Movement                     | Hit Object in Driveway           |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   9   | Date                                    | Car Passenger                           | Vehicle Leaving Driveway         |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   10  | Day of Week                             | Bus/Coach Passenger                     | Hit Object off Driveway          |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   11  | Time                                    | Pedestrian Road  Maintenance Worker     | 1st Point of Impact              |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   12  | Local Authority (District)              | Casualty Type                           | Was Vehicle Left Hand Drive      |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   13  | Local Authority (Highway)               | Casualty Home Area Type                 | Journey Purpose of Driver        |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   14  | 1st Road Class                          |                                         | Sex of Driver                    | 
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   15  | 1st Road Number                         |                                         | Age Band of Driver               |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   16  | Road Type                               |                                         | Engine Capacity                  |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   17  | Speed Limit                             |                                         | Propulsion Code                  |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   18  | Junction Detail                         |                                         | Age of Vehicle                   |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   19  | Junction Control                        |                                         | Driver IMD Decile                |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   20  | 2nd Road Class                          |                                         | Driver Home Area Type            |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   21  | 2nd Road Number                         |                                         |                                  |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   22  | Pedestrian Crossing Human Control       |                                         |                                  |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   23  | Pedestrian Crossing Physical Facilities |                                         |                                  |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   24  | Light Conditions                        |                                         |                                  |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   25  | Weather Conditions                      |                                         |                                  |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   26  | Road Surface Conditions                 |                                         |                                  |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   27  | Special Conditions                      |                                         |                                  |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   28  | Carriage Hazards                        |                                         |                                  |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   29  | Urban or Rural Area                     |                                         |                                  |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   30  | Did Police Officer Attend Scene         |                                         |                                  |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+
|   31  | LSOA of Accident Location               |                                         |                                  |
+-------+-----------------------------------------+-----------------------------------------+----------------------------------+

Most of the values are determined by special codes which
which can be found in the pages of 
:download:`this spreadsheet <./Road-Accident-Safety-Data-Guide-1979-2004.xls>`

Example Job
###########

Let's use what we've learned to answer a quick question.
Between 1974 and 2004 how were there more casualties per
incident in rural or urband accidents?

Our mapper will need to emit a key that represents whether
the accident was rural or urban and the number of casualties
as the value. 

Our reducer will need to sum the casualties for each type of
accident and divide them by the total number of accidents.

Given that the code that tells whether a crash was urban or
rural is stored at index 29 of the accident csv and the 
number of casualties is stored at index 8 our code looks like
this:

.. code-block:: python
    :linenos:

    def mapper(key, value):
      data = key.split(',')
      casualties = data[8]
      urbanOrRural = data[29]
      Wmr.emit(urbanOrRural, casualties)

    def reducer(key, values):
      count = 0
      casualties
      for value in values:
        casualties += int(value)
        count += 1
      avgCasualties = casualties / count 
      Wmr.emit(key, avgCasualties)

.. note::
    Does this reducer look familiar?


Run this job on wmr using cluster path 
``/shared/traffic/Accidents7904.csv`` You should get the following
output:

+----+---------------------+
| 1  | 1.2805146224316546  |
+----+---------------------+
| 2  |  1.5105844913989401 |
+----+---------------------+
| 3  | 1.4071045576407506  |
+----+---------------------+
| -1 | 1.3062582787269292  |
+----+---------------------+
    
A quick glance at the spreadsheet reveals that 1 stands for
Urban, 2 for rural, and 3 for unallocated. -1 means that neither
was reported. It appears that on average rural accidents tend
to involve more casualties.
