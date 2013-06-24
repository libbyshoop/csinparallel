************
Introduction
************

The field of epidemiology is the study of infectious diseases, and mathematical models are often used by epidemiologists to describe or predict the spread of disease. Here we present a simplified model of an infectious disease which is *contagious*, meaning spread from person to person.

The model describes a specific population and the current state of health of each member of the population. These states can include *susceptible* (having the potential to get sick), *infected* (actively sick and able to spread the infection to others), and *recovered* (no longer sick and incapable of getting sick again). Based on these three states of health, the model is called the *SIR model*. 

:note: This implementation does not consider the case where an individual dies from the disease; more sophisticated implementations of the SIR model consider this case along with other factors left out here (such as immunity, treatment, or births or non-epidemic related deaths in the population during the modeled time). You can read about more complex models `here`_.

The population starts out with an initial number infected. Associated with the particular infection being modeled are a *radius of transmission*\ , *contagiousness factor*\ , and *duration of illness*. 

	.. glossary::
     	 radius of transmission
		distance within which a susceptible individual must be from an infected individual to become infected

	.. glossary::
    	 contagiousness factor
		when a susceptible individual is within an infected individual’s radius of transmission, the percent chance that transmission will occur from person to person and the susceptible individual will become infected

	.. glossary::
    	 duration of illness
		the length of time an infected individual remains sick and able to transmit the disease

These factors influence the spread of infectious disease and, as such, are the parameters used to control the mathematical model. We use Monte Carlo simulation techniques (running a simulation over and over again) to approximate this SIR model and thereby gather information on how a disease would progress through a population.

.. _here: http://en.wikipedia.org/wiki/Epidemic_model