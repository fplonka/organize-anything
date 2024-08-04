package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
)

func main() {
	http.HandleFunc("/", serveFile("fragments/skeleton.html"))
	http.HandleFunc("/fragment/", serveFragment)

	fmt.Println("Server is running on http://localhost:8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func serveFile(filename string) http.HandlerFunc {
	fmt.Println("want:", filename)
	return func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, filename)
	}
}

func serveFragment(w http.ResponseWriter, r *http.Request) {
	fragmentID := filepath.Base(r.URL.Path)
	fmt.Println("want:", fragmentID)
	filename := filepath.Join("fragments", fragmentID+".html")

	if _, err := os.Stat(filename); os.IsNotExist(err) {
		http.NotFound(w, r)
		return
	}

	http.ServeFile(w, r, filename)
}
