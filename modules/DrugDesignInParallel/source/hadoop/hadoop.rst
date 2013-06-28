***************
Hadoop Solution
***************

Hadoop is an open-source framework for data-intensive scalable map-reduce computation. Originally developed by Yahoo! engineers and now an Apache project, Hadoop supports petascale computations in a reasonable amount of time (given sufficiently large cluster resources), and is used in numerous production web-service enterprises. The code :download:`dd_hadoop.java <code/dd_hadoop.java>` implements a solution to our problem for the Hadoop map-reduce framework, which is capable of data-intensive scalable computing.  

In our previous examples, we have modified the coding of a map-reduce framework represented by the C++ method ``MR::run()`` in order to create implementations with various parallelization technologies. Hadoop provides a powerful implementation of such a framework, with optimizations for large-scale data, adaptive scheduling of tasks, automated recovery from failures (which will likely occur when using many nodes for lengthy computations), and an extensive system for reusable configuration of jobs. To use Hadoop, one needs only provide ``Map()``, ``Reduce()``, configuration options, and the desired data.  This framework-based strategy makes it convenient for Hadoop programmers to create and launch effective, scalably large computations. 

Therefore, we will compare definitions of ``Map()`` and ``Reduce()`` found in the serial implementation :download:`dd_serial.cpp <code/dd_serial.cpp>` to the corresponding definitions in a Hadoop implementation :download:`dd_hadoop.java <code/dd_hadoop.java>`. The serial implementations for our simplified problem are quite simple:

	.. code-block:: java
		:emphasize-lines: 1,6
		:linenos:

		void MR::Map(const string &ligand, vector<Pair> &pairs) {
			Pair p(Help::score(ligand.c_str(), protein.c_str()), ligand);
			pairs.push_back(p);
		}

		int MR::Reduce(int key, const vector<Pair> &pairs, int index, string &values) {
			while (index < pairs.size() && pairs[index].key == key) {
				values += pairs[index++].val + " ";
			}
			return index;
		}

Here, ``Map()`` has two arguments, a ligand to compare to the target protein and an STL vector ``pairs`` of key-value pairs. A call to ``Map()`` appends a pair consisting of that ligand’s score (as key) and that ligand itself (as value) to the vector ``pairs``. Our ``Reduce()`` function extracts all the key-value pairs from the (now sorted) vector ``pairs`` having a given key (i.e., score). It then appends a string consisting of all those values (i.e., ligands) to an array ``values``\ . The argument ``index`` and the return value are used by ``MR::run()`` in order to manage progress through the vector ``pairs``\ (our multi-threaded implementations have identical ``Map()`` and ``Reduce()`` methods, except that a thread-safe vector type is used for ``pairs``\ ). In brief, ``Map()`` receives ``ligand`` values and produces pairs, and ``Reduce()`` receives pairs and produces consolidated results in ``values``\ .  

In Hadoop, we define the “map” and “reduce” operations as Java methods ``Map.map()`` and ``Reduce.reduce()``\ . Here are definitions of those methods from :download: `dd_hadoop.java <code/dd_hadoop.java>`\ :

	.. code-block:: java
		:emphasize-lines: 1,9
		:linenos:

		public void map(LongWritable key, Text value, OutputCollector<IntWritable, Text> output, Reporter reporter) 
				throws IOException {
			String ligand = value.toString();
			output.collect(new IntWritable(score(ligand, protein)), value);
		}

				...

		public void reduce(IntWritable key, Iterator<Text> values, OutputCollector<IntWritable, Text> output, Reporter reporter) 
				throws IOException {
			String result = new String("");
			while (values.hasNext()) {
				result += values.next().toString() + " ";
			}
			output.collect(key, new Text(result));
		}

In brief, our Hadoop implementation’s ``map()`` receives a key and a value, and produces pairs to the ``OutputCollector`` argument ``output``\ , and ``reduce()`` receives a key and an iterator of values and produces consolidated results in an ``OutputCollector`` argument (also named ``output``\ ). In Hadoop, the values from key-value pairs sent to a particular call of ``reduce()`` are provided in an *iterator* rather than a vector or array, since there may be too many values to hold in memory with very large scale data. Likewise, the ``OutputCollector`` type can handle arbitrarily many key-value pairs.  

Further Notes
#############

- The Hadoop types ``Text``\ , ``LongWritable``\ ,  and ``IntWritable`` represent text and integer values in formats that can be communicated through Hadoop’s framework stages. Also, the method ``OutputCollector.collect()`` adds a key-value pair to an OutputCollector instance like ``output``.

- *Note on scalability:* Our ``reduce()`` method consolidates all the ligands with a given score into a single string (transmitted as ``Text``\ ), but this appending of strings does not scale to very large data. If, for example, trillions of ligand strings are possible, then ``reduce()`` must be revised. For example, one might use a trivial reducer that will produce a fresh key-value pair for each score and ligand, effectively copying key-value pairs to the same key-value pairs. Automatic sorting services provided by Hadoop between the “map” and “reduce” stages will ensure that the output produced by the “reduce” stage is sorted by the ``key`` argument for calls to ``reduce()``. Since those ``key`` arguments are scores for ligands in our application, this automatic sorting by ``key`` makes it simpler to identify the ligands with large scores from key-value pairs produced by that trivial reducer.

Questions for exploration
#########################

- Try running the example :download:`dd_hadoop.java <code/dd_hadoop.java>` on a system with Hadoop installed.  

	- This code does not generate data for the “map” stage, so you will have to produce your own randomly generated ligands, perhaps capturing the output from ``Generate_tasks()`` for one of the other implementations.  

	- Once you have a data set, you must place it where your Hadoop application can find it.  One ordinarily does this by uploading that data to the Hadoop Distributed File System (HDFS), which is typically tuned for handling very large data (e.g., unusually large block size and data stored on multiple disks for fault tolerance).  

	- Rename the source file to ``DDHadoop.java`` (if necessary) before attempting to compile. After compiling the code, packaging it into a `.jar`_ file, and submitting that Hadoop job, you will probably notice that running the Hadoop job takes far more time than any of our other implementations (including sequential), while producing the same results. This is because the I/O overhead used to launch a Hadoop job dominates the computation time for small-scale data. However, with data measured in terabytes of petabytes, it prepares for effective computations in reasonable time (see `Amdahl's law`_).

	- Hadoop typically places the output from processing in a specified directory on the HDFS. By default, if the “map” stage generates relatively few key-value pairs, a single thread/process performs ``reduce()`` calls in the “reduce” stage, yielding a single output file (typically named ``part-00000``).  

- Modify :download:`dd_hadoop.java <code/dd_hadoop.java>` to use a trivial reducer instead of a reducer that concatenates ligand strings. Compare the output generated with a trivial reducer to the output generated by :download:`dd_hadoop.java <code/dd_hadoop.java>`.

- Research the configuration change(s) necessary in order to compute with multiple ``reduce()`` threads/processes at the “reduce” stage. Note that each such thread or process produces its own output file ``part-NNNNN``\ . Examine those output files, and note that they are sorted by the ``key`` argument for ``reduce()`` within each output file.  

- Would it be possible to scale one of our other implementations to compute with terabytes of data in a reasonable amount of time? Consider issues such as managing such large data, number of threads/nodes required for reasonable elapsed time, capacity of data structures, etc. Are some implementations more scalable than others?

- For further ideas, see exercises for other parallel implementations.

.. _.jar: http://en.wikipedia.org/wiki/JAR_(file_format)
.. _Amdahl's law: http://home.wlu.edu/~whaleyt/classes/parallel/topics/amdahl.html

Readings about map-reduce frameworks and Hadoop
###############################################

- `[Dean and Ghemawat, 2004]`_  J. Dean and S. Ghemawat. MapReduce: Simplified data processing on large clusters, 2004. 

- `[Hadoop]`_  Apache Software Foundation. Hadoop. 

- [White, 2011]  T. White, Hadoop:  The definitive guide, O’Reilly, 2nd edition, 2011.  

.. _[Dean and Ghemawat, 2004]: http://labs.google.com/papers/mapreduce.html

.. _[Hadoop]: http://hadoop.apache.org/core/
