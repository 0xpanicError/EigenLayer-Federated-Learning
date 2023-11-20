package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/urfave/cli"

	"github.com/0xpanicError/EigenLayer-Federated-Learning/aggregator"
	"github.com/0xpanicError/EigenLayer-Federated-Learning/core/config"
)

var (
	// Version is the version of the binary.
	Version   string
	GitCommit string
	GitDate   string
)

func main() {

	app := cli.NewApp()
	app.Flags = config.Flags
	app.Version = fmt.Sprintf("%s-%s-%s", Version, GitCommit, GitDate)
	app.Name = "federated-learning-aggregator"
	app.Usage = "Federated Learning Aggregator"
	app.Description = "Service that sends base model parameters to nodes which train and update the model locally."

	app.Action = aggregatorMain
	err := app.Run(os.Args)
	if err != nil {
		log.Fatalln("Application failed.", "Message:", err)
	}
}

func aggregatorMain() error {

	log.Println("Initializing Aggregator")

	agg, err := aggregator.NewAggregator()
	if err != nil {
		return err
	}

	err = agg.Start(context.Background())
	if err != nil {
		return err
	}

	return nil

}
