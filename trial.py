from utils import logger

callbacks = {
    'step' : [ logger.Print() ],
    'epoch': [ logger.Print() ]
}
log = logger.Logger(callbacks)


num_epochs = 2
num_steps_train  = 5
num_steps_test  = 5

for e in range(num_epochs):
    
    for s in range(num_steps_train):
        log.add_step('train')
    
    for s in range(num_steps_test):
        log.add_step('test')
    
    log.add_epoch()

