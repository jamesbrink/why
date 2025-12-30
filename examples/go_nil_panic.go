// Nil pointer dereference panic in Go
package main

import "fmt"

type Config struct {
	DatabaseURL string
	MaxConns    int
}

type App struct {
	config *Config
}

func NewApp() *App {
	// Oops, forgot to initialize config
	return &App{}
}

func (a *App) Connect() error {
	// This will panic - config is nil!
	fmt.Printf("Connecting to %s with %d connections\n",
		a.config.DatabaseURL,
		a.config.MaxConns)
	return nil
}

func main() {
	app := NewApp()
	if err := app.Connect(); err != nil {
		fmt.Println("Failed to connect:", err)
	}
}
