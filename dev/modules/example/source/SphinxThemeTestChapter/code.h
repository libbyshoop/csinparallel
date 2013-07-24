/*
    find_infected()
        Each process determines its infected x locations 
        and infected y locations
*/
void find_infected(struct global_t *global, struct our_t *our)
{
    // counter to keep track of person in our struct
    int our_person1;

    int our_current_infected_person = 0;
    for(our_person1 = 0; our_person1 <= our->our_number_of_people - 1; 
        our_person1++)
    {
        if(our->our_states[our_person1] == INFECTED)
        {
            our->our_infected_x_locations[our_current_infected_person] = 
            our->our_x_locations[our_person1];
            our->our_infected_y_locations[our_current_infected_person] =
            our->our_y_locations[our_person1];
            our_current_infected_person++;
        }
    }
}