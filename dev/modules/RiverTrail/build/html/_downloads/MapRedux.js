<<<<<<< HEAD
/* Super lazy version of random wordcount algo for map-reduce */ 
/* requires the rivertrail plugin for firefox, and a copy of the opencl sdk if you are on windows */
/* see https://github.com/RiverTrail/RiverTrail/wiki */


/* Word count */
var p = function(str){
	//Declare some vars to use down below
    var holder,words, it=0,str1, counts,reduced, totalWords;
	
	//Masage the input
	str1 = str.toLowerCase();
	str = str1.replace(/[.,?()-=+]/g, "");
    words = str.split(" "); /* words is an array now of the input, stripped of bad-chars and all in lowercase */
	totalWords = words.length; // total number of words 
	
	//words is an array of words we want to use for our job
    holder = uniqueArr(words);
	//Holder now has 4 keys:
		//outArr:The input array in a numeric representation using the keys in words2
		//words: which is an object containing all of the mappings for string keys to numeric keys
		//words2: an array mapping numeric key to string value
		//num: the number of unique elements
	
	var paraString = new ParallelArray(holder.outArr); //lets make a ParallelArray

	counts = paraString.map(function(){return 1}); //map a 1 for each word
	
	/* use the scatter function to map the words by numeric key to a count of their occurence*/
	reduced = counts.scatter(paraString,0, function plus(a,b){return a+b}, holder.num);
	//console.log('reduced', reduced);
	
	
	
	var rstr = reduced.toString();
	var rArr = rstr.substring(1,rstr.length-1).split(',');
	//console.log(rArr);
	
	/* create output string */
	rstr = '<h2>Output: </h2><p>Total Words:'+totalWords+'</p>';
	/* <p> is an html element open tag, </p> closes the tag*/
	/* You can concat to the output string using +=, javascript vars can be added to strings like in the line above */
	
	rstr +='<table>'
	for (it=0; it<rArr.length;it++){
		rstr += '<tr><td>' + holder.words2[it] + '</td><td>' + rArr[it] +'</td></tr>';
	};
	rstr +='</table>';
	/* Add it to the output div*/
	document.getElementById('output').innerHTML = rstr;
	

    
};
/* Helper function for wordcount creates all the pieces for the parallel operation*/
var uniqueArr = function(inArr){
	var keys, it=0, itt, ittt=0;
	var words1 = {}, words2=[], words3=[], retobj;
	for (it =0; it <inArr.length;it++){ /* unique words hash creation */
		if(!words1.hasOwnProperty(inArr[it])){ 
			words1[inArr[it]]=ittt; 
			ittt++; 
		};
		words3[it] = words1[inArr[it]]; //inArr as the keys in words obj
	};
	console.log('words3',words3);
	//words1 is the array object of unique words, and their number
	//words one, keys are strings, values are the numeric key
	console.log('num-eles', ittt);
	console.log("words1", words1);

	
	for(itt in words1){ //unique words to array, numeric keys 
		words2[words1[itt]] = itt;
	}; 
	console.log("words2", words2);
	retobj = {'outArr':words3, 'words':words1, 'words2':words2, 'num':ittt};
	return retobj;
}

/* average value function for movielense formatted data */
var q = function(input){
	var bigArr=[], smArr=[], ratings=[];
	var unique={}, itt=0, output='',output1='', timah;
	var i,j=0, ita;
	console.time("timer"); /* Starts timer 'timer' */
	/* Split on newlines and tabs into an array*/
	bigArr = input.split(/[\n\t]/);
	
	/* If this safely handled strings, I would be keeping the date field, or dropping it with map */ 
	/*drop the date field */
	for(i=0; i<bigArr.length-4;i+=4,j+=3){
		//console.log(bigArr[i],'		',bigArr[i+1],'		 ',bigArr[i+2],'	 ',bigArr[i+3]);
		/* note the order of the array we split things into*/ 
		/* User-id		Movie-id		rating		date*/
		
		smArr[j] = bigArr[i]; //copy id into a new array
		smArr[j+1] = bigArr[i+1];
		smArr[j+2] = (bigArr[i+2]).toString().substr(0,1); /*Float math doesn't work well for average, trunking to int */
		
		
		if(!unique.hasOwnProperty(smArr[j])){ /*big time hack to get unique ids, hash by id, storing rating */
			unique[smArr[j]]=((bigArr[i+2]).toString().substr(0,1)).toString();
		}else{
			unique[smArr[j]] = (unique[smArr[j]]).toString()+','+((bigArr[i+2]).toString().substr(0,1)).toString();
		};
		
	};	
	
	/* We have an object(accessed like an array) of unique ids, holding strings of ratings, lets make parallel arrays and do some averaging*/
	for (ita in unique){ /* for each id, make an array with split, then make a parallelArray, then reduce*/
		ratings = unique[ita].split(',');
		
		if(unique.hasOwnProperty(ita)){ //sanity guard, it's a for-in loop after all
			unique[ita+"p"] = new ParallelArray(ratings); //Might as well just add the parallel arrays to the object
		};
		if(unique.hasOwnProperty(ita+"p")){ //sanity guard, it's a for-in loop after all
			/* unique[Numberp], ita is a number, added the string p for the name: now holds a parallel array of the ratings for that id*/
			/* Need to use parseInt since we added as string */
			unique[ita+"av"] = unique[ita+"p"].reduce(function sumr(a,b){return parseInt(a,10)+parseInt(b,10)}) / ratings.length;
			/* Add a row to the output table*/
			output+='<tr><td>'+ita+'</td><td>'+unique[ita+"av"]+'</td></tr>'; 
		};
		
	};
	//console.log(unique); //Does this hold what I thought it would?
	
	/* End timer, add timer to output string, combine all to format*/
	timah = console.timeEnd("timer");
	
	/*To be properly formatted, each row needs the same number of elements so if you add an extra element in the loop, add one to the header row */
	output1+='<h2>Output</h2><p>'+timah+'ms</p><table><tr><td>ID</td><td>Average Rating</td></tr>';
	//console.log(timer)
	output +='</table>';
	output = output1+output;
	
	//add output to page
	document.getElementById('output').innerHTML = output;
	
	
	
	
}
=======
/* Super lazy version of random wordcount algo for map-reduce */ 
/* requires the rivertrail plugin for firefox, and a copy of the opencl sdk if you are on windows */
/* see https://github.com/RiverTrail/RiverTrail/wiki */


/* Word count */
var p = function(str){
	//Declare some vars to use down below
    var holder,words, it=0,str1, counts,reduced, totalWords;
	
	//Masage the input
	str1 = str.toLowerCase();
	str = str1.replace(/[.,?()-=+]/g, "");
    words = str.split(" "); /* words is an array now of the input, stripped of bad-chars and all in lowercase */
	totalWords = words.length; // total number of words 
	
	//words is an array of words we want to use for our job
    holder = uniqueArr(words);
	//Holder now has 4 keys:
		//outArr:The input array in a numeric representation using the keys in words2
		//words: which is an object containing all of the mappings for string keys to numeric keys
		//words2: an array mapping numeric key to string value
		//num: the number of unique elements
	
	var paraString = new ParallelArray(holder.outArr); //lets make a ParallelArray

	counts = paraString.map(function(){return 1}); //map a 1 for each word
	
	/* use the scatter function to map the words by numeric key to a count of their occurence*/
	reduced = counts.scatter(paraString,0, function plus(a,b){return a+b}, holder.num);
	//console.log('reduced', reduced);
	
	
	
	var rstr = reduced.toString();
	var rArr = rstr.substring(1,rstr.length-1).split(',');
	//console.log(rArr);
	
	/* create output string */
	rstr = '<h2>Output: </h2><p>Total Words:'+totalWords+'</p>';
	/* <p> is an html element open tag, </p> closes the tag*/
	/* You can concat to the output string using +=, javascript vars can be added to strings like in the line above */
	
	rstr +='<table>'
	for (it=0; it<rArr.length;it++){
		rstr += '<tr><td>' + holder.words2[it] + '</td><td>' + rArr[it] +'</td></tr>';
	};
	rstr +='</table>';
	/* Add it to the output div*/
	document.getElementById('output').innerHTML = rstr;
	

    
};
/* Helper function for wordcount creates all the pieces for the parallel operation*/
var uniqueArr = function(inArr){
	var keys, it=0, itt, ittt=0;
	var words1 = {}, words2=[], words3=[], retobj;
	for (it =0; it <inArr.length;it++){ /* unique words hash creation */
		if(!words1.hasOwnProperty(inArr[it])){ 
			words1[inArr[it]]=ittt; 
			ittt++; 
		};
		words3[it] = words1[inArr[it]]; //inArr as the keys in words obj
	};
	console.log('words3',words3);
	//words1 is the array object of unique words, and their number
	//words one, keys are strings, values are the numeric key
	console.log('num-eles', ittt);
	console.log("words1", words1);

	
	for(itt in words1){ //unique words to array, numeric keys 
		words2[words1[itt]] = itt;
	}; 
	console.log("words2", words2);
	retobj = {'outArr':words3, 'words':words1, 'words2':words2, 'num':ittt};
	return retobj;
}

/* average value function for movielense formatted data */
var q = function(input){
	var bigArr=[], smArr=[], ratings=[];
	var unique={}, itt=0, output='',output1='', timah;
	var i,j=0, ita;
	console.time("timer"); /* Starts timer 'timer' */
	/* Split on newlines and tabs into an array*/
	bigArr = input.split(/[\n\t]/);
	
	/* If this safely handled strings, I would be keeping the date field, or dropping it with map */ 
	/*drop the date field */
	for(i=0; i<bigArr.length-4;i+=4,j+=3){
		//console.log(bigArr[i],'		',bigArr[i+1],'		 ',bigArr[i+2],'	 ',bigArr[i+3]);
		/* note the order of the array we split things into*/ 
		/* User-id		Movie-id		rating		date*/
		
		smArr[j] = bigArr[i]; //copy id into a new array
		smArr[j+1] = bigArr[i+1];
		smArr[j+2] = (bigArr[i+2]).toString().substr(0,1); /*Float math doesn't work well for average, trunking to int */
		
		
		if(!unique.hasOwnProperty(smArr[j])){ /*big time hack to get unique ids, hash by id, storing rating */
			unique[smArr[j]]=((bigArr[i+2]).toString().substr(0,1)).toString();
		}else{
			unique[smArr[j]] = (unique[smArr[j]]).toString()+','+((bigArr[i+2]).toString().substr(0,1)).toString();
		};
		
	};	
	
	/* We have an object(accessed like an array) of unique ids, holding strings of ratings, lets make parallel arrays and do some averaging*/
	for (ita in unique){ /* for each id, make an array with split, then make a parallelArray, then reduce*/
		ratings = unique[ita].split(',');
		
		if(unique.hasOwnProperty(ita)){ //sanity guard, it's a for-in loop after all
			unique[ita+"p"] = new ParallelArray(ratings); //Might as well just add the parallel arrays to the object
		};
		if(unique.hasOwnProperty(ita+"p")){ //sanity guard, it's a for-in loop after all
			/* unique[Numberp], ita is a number, added the string p for the name: now holds a parallel array of the ratings for that id*/
			/* Need to use parseInt since we added as string */
			unique[ita+"av"] = unique[ita+"p"].reduce(function sumr(a,b){return parseInt(a,10)+parseInt(b,10)}) / ratings.length;
			/* Add a row to the output table*/
			output+='<tr><td>'+ita+'</td><td>'+unique[ita+"av"]+'</td></tr>'; 
		};
		
	};
	//console.log(unique); //Does this hold what I thought it would?
	
	/* End timer, add timer to output string, combine all to format*/
	timah = console.timeEnd("timer");
	
	/*To be properly formatted, each row needs the same number of elements so if you add an extra element in the loop, add one to the header row */
	output1+='<h2>Output</h2><p>'+timah+'ms</p><table><tr><td>ID</td><td>Average Rating</td></tr>';
	//console.log(timer)
	output +='</table>';
	output = output1+output;
	
	//add output to page
	document.getElementById('output').innerHTML = output;
	
	
	
	
}
>>>>>>> 8766cf121e46c568468d697515e36d67e1be51f7
