*******************************
Context of Map-Reduce Computing
*******************************

* The use of LISP's map and reduce functions to solve computational problems probably dates from the 1960s -- very early in the history of programming languages

* In 2004, Google published their adaptation of the map-reduce strategy for data-intensive scalable computing (DISC) on large clusters. Their implementation, called **MapReduce**, incorporates features automatically to split up enormous (e.g., multiple *petabytes*) data sets, schedule the mapper and reducer processes, arrange for those processes always to operate on local data for performance efficiency, and recover from faults such as computers or racks crashing.

* MapReduce, together with the page rank algorithm, gave Google the competitive combination it needed to become the most popular search engine (approximately 2/3 of the market at present). Google proceeded to apply map-reduce techniques to everything from ad placement to maps and document services.

* Google's MapReduce is proprietary software. But Yahoo! created the Hadoop implementation of this map-reduce strategy for clusters as an Apache Software Foundation open-source project. Consequently, Hadoop is used not only at Yahoo!, but at numerous other web service companies, and is available for use at colleges and universities.

* *Future systems:* (1) Strategies such as map-reduce that enable programmers to provide relatively simple code segments and reuse code for synchronization, fault tolerance, etc., are a target for forthcoming systems (*View from Berkeley*, 2006). (2) Future systems are likely to consist of multiple heterogeneous cores, programmed using functional programming techniques (Michael Wrinn, Intel, keynote speech at SIGCSE 2010).
