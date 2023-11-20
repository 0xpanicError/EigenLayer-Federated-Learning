package aggregator

import (
	"context"
	"math/big"
	"sync"
	"time"

	"encoding/json"
    "crypto/sha256"

	"github.com/Layr-Labs/eigensdk-go/logging"
	"github.com/Layr-Labs/eigensdk-go/signer"
	"github.com/ethereum/go-ethereum/accounts/abi/bind"

	sdkclients "github.com/Layr-Labs/eigensdk-go/chainio/clients"
	sdkelcontracts "github.com/Layr-Labs/eigensdk-go/chainio/elcontracts"
	"github.com/Layr-Labs/eigensdk-go/services/avsregistry"
	blsagg "github.com/Layr-Labs/eigensdk-go/services/bls_aggregation"
	pubkeycompserv "github.com/Layr-Labs/eigensdk-go/services/pubkeycompendium"
	sdktypes "github.com/Layr-Labs/eigensdk-go/types"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Aggregator struct {}

type convnet struct {
	g                  *gorgonia.ExprGraph
	w0, w1, w2, w3, w4 *gorgonia.Node // weights. the number at the back indicates which layer it's used for
	d0, d1, d2, d3     float64        // dropout probabilities

	out *gorgonia.Node
}

type parameters struct {
	g   *gorgonia.ExprGraph
	x   *gorgonia.Tensor
	y   *gorgonia.Matrix
	m   *convnet
}

func NewAggregator() (*Aggregator, error) {}

func (a *Aggregator) Start(ctx context.Context) error {
	agg.logger.Infof("Starting aggregator.")
	agg.logger.Infof("Starting aggregator rpc server.")
	go agg.startServer(ctx)

	flag.Parse()
	parseDtype()
	rand.Seed(1337)

	// intercept Ctrl+C
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	doneChan := make(chan bool, 1)

	var inputs, targets tensor.Tensor
	var err error

	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()

	numExamples := inputs.Shape()[0]
	bs := *batchsize
	// todo - check bs not 0

	// set a random set of base Model parameters
	if err := inputs.Reshape(numExamples, 1, 28, 28); err != nil {
		log.Fatal(err)
	}
	g := gorgonia.NewGraph()
	x := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(bs, 1, 28, 28), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(bs, 10), gorgonia.WithName("y"))
	m := newConvNet(g)

	baseParams = &parameters{
		g:   g,
		x:   x,
		y:   y,
		m:   m,
	}

	_ = agg.sendBaseParameters(baseParams)

	for {
		select {
		case <-ctx.Done():
			return nil
		case flAggResp := <-agg.flAggregationService.GetResponseChannel():
			agg.logger.Info("Received response from flAggregationService", "flAggServiceResp", flAggServiceResp)
			agg.sendAggregatedResponseToContract(flAggServiceResp)
		}
	}
}

func newConvNet(g *gorgonia.ExprGraph) *convnet {
	w0 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(32, 1, 3, 3), gorgonia.WithName("w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w1 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(64, 32, 3, 3), gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w2 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(128, 64, 3, 3), gorgonia.WithName("w2"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w3 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(128*3*3, 625), gorgonia.WithName("w3"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w4 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(625, 10), gorgonia.WithName("w4"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	return &convnet{
		g:  g,
		w0: w0,
		w1: w1,
		w2: w2,
		w3: w3,
		w4: w4,

		d0: 0.2,
		d1: 0.2,
		d2: 0.2,
		d3: 0.55,
	}
}

func (a *Aggregator) sendBaseParameters(baseParams *parameters) error {
	agg.logger.Info("Aggregator sending new task", "parameters", parameters)

	// Send hash of baseParameters to the task manager contract
	// need to find a new mthod to send param info as solidity does not support tensor data
	jsonData, _ := json.Marshal(baseParams)
    hash := sha256.Sum256(jsonData)

	agg.logger.Info("Aggregator sending new task", "hash", hash)
	// Send hash of baseParameters to the task manager contract
	newTask, taskIndex, err := agg.avsWriter.SendNewTaskUpdateParameters(context.Background(), hash, types.QUORUM_THRESHOLD_NUMERATOR, types.QUORUM_NUMBERS)
	if err != nil {
		agg.logger.Error("Aggregator failed to send parameters", "err", err)
		return err
	}

	quorumThresholdPercentages := make([]uint32, len(newTask.QuorumNumbers))
	for i, _ := range newTask.QuorumNumbers {
		quorumThresholdPercentages[i] = newTask.QuorumThresholdPercentage
	}
	return nil
}
