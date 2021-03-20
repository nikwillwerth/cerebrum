#include "cerebrum/layers/Dense.h"
#include "cerebrum/layers/Input.h"
#include "cerebrum/layers/Softmax.h"
#include "cerebrum/layers/ReLU.h"
#include "cerebrum/layers/Conv2D.h"
#include "cerebrum/Model.h"
#include "cerebrum/utils/NetIO.h"

int main()
{
    size_t batchSize = 2;

    auto *input   = new Input({batchSize, 28, 28, 1});
    auto *conv    = new Conv2D(input, 10, 28, 1, "valid");
    auto *softmax = new Softmax(conv);

    auto *model = new Model();
    model->compile(input);
    model->train(batchSize, 100000000);

    return 0;
}
