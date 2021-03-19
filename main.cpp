#include "cerebrum/layers/Dense.h"
#include "cerebrum/layers/Input.h"
#include "cerebrum/layers/Softmax.h"
#include "cerebrum/layers/ReLU.h"
#include "cerebrum/layers/Conv2D.h"
#include "cerebrum/optimizers/SGD.h"
#include "cerebrum/Model.h"

int main()
{
    size_t batchSize = 2;

    auto *input    = new Input({batchSize, 28, 28, 1});
    auto *convOne  = new Conv2D(input, 8, 3, 1, "same");
    auto *reluOne  = new ReLU(convOne);
    auto *convTwo  = new Conv2D(reluOne, 8, 3, 1, "same");
    auto *reluTwo  = new ReLU(convTwo);
    auto *dense    = new Dense(reluTwo, 2);
    auto *softmax  = new Softmax(dense);

    auto *model = new Model();
    model->compile(input);
    model->train(batchSize, 1000);

    return 0;
}
