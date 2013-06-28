package main

import (
        "flag"
        "fmt"
        "rand"
        "runtime"
        "sort"
)

var (
        nLigands  = flag.Int("n", 120, "number of ligands")
        maxLigand = flag.Int("len", 7, "maximum length of a ligand")
        protein   = flag.String("protein",
                "the cat in the hat wore the hat to the cat hat party",
                "protein to use")
        nCPU = flag.Int("cpu", 1, "number of CPUs to use")
)

type Pair struct {
        score int
        ligand []byte
}

type PairArray []Pair

func main() {
        flag.Parse()
        runtime.GOMAXPROCS(*nCPU) // Set number of OS threads to use.

        ligands := make(chan []byte, 1024)
        go func() {
                for i := 0; i < *nLigands; i++ {
                        len := rand.Intn(*maxLigand) + 1
                        l := make([]byte, len)
                        for j := 0; j < len; j++ {
                                l[j] = byte('a' + rand.Intn(26))
                        }
                        ligands <- l
                }
                close(ligands)
        }()

        pairs := make(chan Pair, 1024)
        for i := 0; i < *nCPU; i++ {
                go func() {
                        p := []byte(*protein)
                        for l := range ligands {
                                pairs <- Pair{score(l, p), l}
                        }
                }()
        }

        sorted := make(PairArray, *nLigands)
        for i := 0; i < *nLigands; i++ {
                sorted[i] = <-pairs
        }
        sort.Sort(&sorted)

        var results PairArray
        for i := 0; i < len(sorted); {
                s := sorted[i].score
                var l []byte
                for ; i < len(sorted) && s == sorted[i].score; i++ {
                        l = append(append(l, sorted[i].ligand...), ' ')
                }
                results = append(results, Pair{s, l})
        }

        fmt.Printf("maximal score is %d, achieved by ligands %s\n",
                results[0].score, string(results[0].ligand))
}

func score(str1, str2 []byte) int {
        if len(str1) == 0 || len(str2) == 0 {
                return 0
        }
        // Both argument strings non-empty.
        if str1[0] == str2[0] {
                return 1 + score(str1[1:], str2[1:])
        }
        // First characters do not match.
        s1 := score(str1, str2[1:])
        s2 := score(str1[1:], str2)
        if s1 > s2 {
                return s1
        }
        return s2
}

func (a *PairArray) Len() int {
        return len(*a)
}

func (a *PairArray) Less(i, j int) bool {
        return (*a)[i].score > (*a)[j].score
}

func (a *PairArray) Swap(i, j int) {
        t := (*a)[i]
        (*a)[i] = (*a)[j]
        (*a)[j] = t
}