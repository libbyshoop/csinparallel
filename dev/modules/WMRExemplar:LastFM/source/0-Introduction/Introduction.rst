########################
The Million Song Dataset
########################

Last.fm is a popular music recommendation website that tracks
what it's users listen to and then suggests similar songs.
It provides an API which can be used to retrieve metadata about
specific songs. 

Researchers at Columbia's LabRosa used this api to generate a
dataset containing 1,000,000 songs from 44,745 artists. For our
purposes this data has been split into 3 different tab seperated
files.

Each file has a different prefix which hadoop passes to the 
mapper as a key. The rest of the data is passed to the mapper as
the value. Each file contains several fields so it is necessary
to call value.split('\t') to access them.

- **/shared/lastfm/similars.tsv** is prefixed with "similar". The
  first value of the data is the id of an artist and the rest
  of the values are the ids of similar artists. The list of
  similar artists varies in size. If artist A's list contains B
  and C there is no guarantee that B's list contains C or A.
  The list of similar artists varies in length and may be empty.

- **/shared/lastfm/terms.tsv** is prefixed with "term". The first
  value of the data is an artist id and the rest of the values
  are comma seperated triplets representing terms associated
  with the artist. They can be seperated by calling 
  term.split(',')
    - The first value is the term itself. It may be a genres like
      "rock" or "pop" or a descriptor like "london"
    - The second value is the frequency with which that term is
      used in reference to the artist, it is a float from 0-1
    - The last value is the weight of the term which provides a
      a measure of how well a given term is to describes the
      an artist. For example 'rock' is frequently used to
      describe the Beatles, but "british invasion" is more
      descriptive so it has a higher weight. The weight is a 
      float from 0-1
  there are a variable number of terms associated with each 
  artist and there may be none.

- **/shared/lastfm/metadata.tsv** is prefixed with "metadata". 
  The data contains 25 different fields of metadata about a given
  song explained in the following chart. These feilds may be null

+-------+---------------------------+-------------------------------------------+
| index | value                     | Description                               |
+=======+===========================+===========================================+
|  0    | track id                  | String                                    |
+-------+---------------------------+-------------------------------------------+
|  1    | title                     | String                                    |
+-------+---------------------------+-------------------------------------------+
|  2    | release name (album)      | String                                    |
+-------+---------------------------+-------------------------------------------+
|  3    | year                      | Int                                       |
+-------+---------------------------+-------------------------------------------+
|  4    | artist name               | String                                    |
+-------+---------------------------+-------------------------------------------+
|  5    | artist id                 | String                                    |
+-------+---------------------------+-------------------------------------------+
|  6    | artist familiarity        | Float 0-1 How well known an aritst is     |
+-------+---------------------------+-------------------------------------------+
|  7    | artist hotttnesss         | Float 0-1 Current popularity of an artist |
+-------+---------------------------+-------------------------------------------+
|  8    | artist latitude           | Float                                     |
+-------+---------------------------+-------------------------------------------+
|  9    | artist longitude          | Float                                     |
+-------+---------------------------+-------------------------------------------+
|  10   | artist location           | String                                    |
+-------+---------------------------+-------------------------------------------+
|  11   | hotttnesss                | Float 0-1 current popularity of a song    |
+-------+---------------------------+-------------------------------------------+
|  12   | danceablity               | Float 0-1                                 |
+-------+---------------------------+-------------------------------------------+
|  13   | duration                  | Float number of seconds in a song         |
+-------+---------------------------+-------------------------------------------+
|  14   | energy                    | Float 0-1                                 |
+-------+---------------------------+-------------------------------------------+
|  15   | loudness                  | Float 0-1                                 |
+-------+---------------------------+-------------------------------------------+
|  16   | end of fade in            | Float                                     |
+-------+---------------------------+-------------------------------------------+
|  17   | start of fade out         | Float                                     |
+-------+---------------------------+-------------------------------------------+
|  18   | tempo                     | Float tempo in beats per minute           |
+-------+---------------------------+-------------------------------------------+
|  19   | time signature            | Int number of beats per measure           |
+-------+---------------------------+-------------------------------------------+
|  20   | time signature confidence | Float 0-1 confidence in the above number  |
+-------+---------------------------+-------------------------------------------+
|  21   | mode                      | 1 for major 0 for minor                   |
+-------+---------------------------+-------------------------------------------+
|  22   | mode confidence           | Float 0-1 confidence in the above number  |
+-------+---------------------------+-------------------------------------------+
|  23   | key                       | Int C=0, C#=1, D=2...                     |
+-------+---------------------------+-------------------------------------------+
|  24   | key confidence            | Float 0-1 confidence in the above number  |
+-------+---------------------------+-------------------------------------------+
