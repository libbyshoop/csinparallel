package lab.spider;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

/**
 * Downloads web page content starting with a starting url.
 * If the spider encounters links in the content, it downloads
 * those as well.
 * 
 * Steps:
 * 1. Complete the processPage method.  One TestSpider unit tests should pass.
 * 2. Complete the crawl() method.  Both TestSpider unit tests should pass.
 *  
 * @author shilad
 *
 */
public class CompletedSpider {
	/**
	 * Urls waiting to be scraped.  The "work" left to do.
	 */
	private Queue<String> work = new LinkedList<String>();
	
	/**
	 * Keeps track of counts for each url.
	 */
	private AllWordsCounter urlCounter = new AllWordsCounter();
	
	/**
	 * Maximum number of urls that should be scraped.
	 */
	private int maxUrls = 100;
	
	/**
	 * URLs that have already been retrieved.
	 */
	private List<String> finished = new ArrayList<String>();
	
	/**
	 * Helps download and parse the web pages.
	 */
	private HttpHelper helper = new HttpHelper();
	
	/**
	 * Creates a new spider that will crawl at most maxUrls.
	 * @param maxUrls
	 */
	public CompletedSpider(int maxUrls) {
		this.maxUrls = maxUrls;
	}
	
	/**
	 * Crawls at most maxUrls starting with beginningUrl.
	 * @param beginningUrl
	 */
	public void crawl(String beginningUrl) {
		work.add(beginningUrl);
		
		// TODO: While there is remaining work and we haven't
		// reach the maximum # of finished urls, process
		// the next unfinshed url.  After processing, mark
		// it as finished.
		while (!work.isEmpty() && finished.size() < maxUrls) {
			String url = work.remove();
			if (!finished.contains(url)) {
				processPage(url);
				finished.add(url);
			}
		}
	}
	
	/**
	 * Retrieves content from a url and processes that content. 
	 * @param baseUrl
	 * @param html
	 */
	public void processPage(String url) {
		String html = helper.retrieve(url);
		
		// TODO: extract all the links from the url
		// For each link that isn't an image, increment the
		// count for the link and queue up the link for future scraping.
		// HINT: Take a look at the helper class
		for (String url2 : helper.extractLinks(url, html)) {
			if (!helper.isImage(url2)) {
				urlCounter.countWord(url2);
				work.add(url2);
			}
		}
	}
	
	/**
	 * Returns the number of times the spider encountered
	 * links to each url.  The url are returned in increasing
	 * frequency order.
	 * 
	 * @return
	 */
	public WordCount[] getUrlCounts() {
		return urlCounter.getCounts();
	}
	
	/**
	 * These getters should only be used for testing.
	 */
	Queue<String> getWork() { return work; }
	List<String> getFinished() { return finished; }
}
