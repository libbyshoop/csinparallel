###########################
Working with Multiple files
###########################

The sample question from the last section was fairly simple
to answer because all of the data could be found in one 
file. However data is often split between files, making it 
harder to process. 

Take this question for instance: are taxis more likely to 
get into crashes on the weekend?

Taxi Crashes
############

To answer this question we will need to access the day of
week data at accidents[10] and the vehicle type data at
vehicles[2] (codes 8 and 108 represent taxis). However those 
two bits of data are in two seperate files so we'll need some
way to cross reference them. We'll do that with the accident
index stored at accidents[0] and vehicles[0]

This also means that we'll need to access multiple files during
a single job. Luckily WMR makes this easy for us. If we enter
a folder into the cluster path, it will use all the files in
that folder has input. 

However we still need to be able to tell if a mapper key came
from the accidents file or the vehicles file. We can do this 
by looking at the length of the data list. The Vehicles file
has 21 peices of information while the Accidents file has 32.
Armed with this information we can write a mapper and a reducer
that will filter out accidents based on whether they involved
a taxi. Run :download:`this code<taxiMapper.py>` using Cluster Path ``/shared/traffic``

.. code-block:: python
    :linenos:

    def mapper(key, value):
        data = key.split(',')
        if len(data) == 21:                 #vehicle data
            if data[2] in ('8', '108'):     #codes for taxis
                Wmr.emit(data[0], "taxi")
        elif len(data) == 32:               #accident data
            Wmr.emit(data[0], data[10])

This mapper checks to see whether input came from accident data
or vehicle data. Then, if it was accident data, it emits the
day of the week that the accident occured on. If it came from 
the vehicles data then it emits a message if a vehicle involved 
was a taxi. 

Our :download:`reducer<taxiReducer.py>` takes that output and emits a list of accident
indices and the day of the week that they occured on.

.. code-block:: python
    :linenos:

    def reducer(key, values):
        isTaxi = False
        dayOfWeek = ""
        for value in values:
            if value == "taxi":
                isTaxi = True
            else:
                dayOfWeek = value
        Wmr.emit(dayOfWeek, key)

This works because only one day of week value is emitted per
accident index and while there can be more than one taxi
involved in a given crash.

But we're not done yet. We simply have list of crashes and
a list of the days on which they occured. We still need to 
count them.

We can this by using the output of the last job to run a new
job. Just hit the use output button at the top
or bottom of the page.

Our mapper will recieve days of the week as keys and ones as the values. 
We just need to feed these straight into a 
:download:`counting reducer<countingReducer.py>`
by using what's known as the :download:`identity mapper<idMapper.py>`
our code is as follows:

.. code-block:: python
    :linenos:

    def mapper(key, value):
        Wmr.emit(key, value)

.. code-block:: python
    :linenos:
  

    def reducer(key, values)
        count = 0
        for value in values:
            count += int(value)
        emit(key, count)

After submitting the job on WMR we get the following output:

+---+---------+
| 1 | 693847  |
+---+---------+
| 2 | 873422  |
+---+---------+
| 3 | 877086  |
+---+---------+
| 4 | 890605  |
+---+---------+
| 5 | 934161  |
+---+---------+
| 6 | 1058859 |
+---+---------+
| 7 | 896218  |
+---+---------+

Code 1 is Sunday, code 2 is Monday etc. So it looks like
Taxis get into the most accidents on Fridays, a fairly high
number on Saturdays, but very few on Sundays.

Challenges
##########

Use the techniques you've learned to answer the following 
questions, or come up with your own:

- Are male drivers more likely to injure other males? You
  will need the following fields: Sex of the driver - 
  Vehicles[14], Sex of casualty - Casualties[4] in both
  cases 1 is male 2 is female 3 is unknown and -1 is 
  missing data.

- What is the average severity of a crash in which at
  least one vehicle overturned? If vehicles[7] = 2, 5, or
  4 the vehicle overturned. The severity of an accident
  is Accidents[6] and ranges from 1-3, 1 being the most
  serious.

- Are trucks more deadly than vans?

- Create a graph showing the number of traffic accidents
  at each hour of the day. If you're feeling adventurous
  seperate it out by day and hour.
  
- Devise some of your own questions to ask of this data.
