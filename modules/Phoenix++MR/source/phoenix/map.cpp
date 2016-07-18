    void map(data_type const& s, map_container& out) const
    {
        for (uint64_t i = 0; i < s.len; i++)
        {
            s.data[i] = toupper(s.data[i]);
        }

        uint64_t i = 0;
        while(i < s.len)
        {            
            while(i < s.len && (s.data[i] < 'A' || s.data[i] > 'Z'))
                i++;
            uint64_t start = i;
            while(i < s.len && ((s.data[i] >= 'A' && s.data[i] <= 'Z') || s.data[i] == '\''))
                i++;
            if(i > start)
            {
                s.data[i] = 0;
                wc_word word = { s.data+start };
                emit_intermediate(out, word, 1);
            }
        }
    }
