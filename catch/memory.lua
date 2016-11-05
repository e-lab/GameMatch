
-- memory for experience replay:
local function Memory(maxMemory, discount)
    local memory = {}

    -- Appends the experience to the memory.
    function memory.remember(memoryInput)
        table.insert(memory, memoryInput)
        if (#memory > maxMemory) then
            -- Remove the earliest memory to allocate new experience to memory.
            table.remove(memory, 1)
        end
    end

    function memory.getBatch(model, batchSize, nbActions, nbStates,useGpu)

        -- We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
        -- batch we can (at the beginning of training we will not have enough experience to fill a batch).
        local memoryLength = #memory
        local chosenBatchSize = math.min(batchSize, memoryLength)

        local inputs = torch.zeros(chosenBatchSize, nbStates)
        local targets = torch.zeros(chosenBatchSize, nbActions)
        if useGpu then inputs, targets = inputs:cuda(), targets:cuda() end

        --Fill the inputs and targets up.
        for i = 1, chosenBatchSize do
            -- Choose a random memory experience to add to the batch.
            local randomIndex = torch.random(1, memoryLength)
            local memoryInput = memory[randomIndex]
            if useGpu then memoryInput.inputState = memoryInput.inputState:cuda() end
            local target = model:forward(memoryInput.inputState)

            --Gives us Q_sa, the max q for the next state.
            if useGpu then memoryInput.nextState = memoryInput.nextState:cuda() end
            local nextStateMaxQ = torch.max(model:forward(memoryInput.nextState), 1)[1]
            if (memoryInput.gameOver) then
                target[memoryInput.action] = memoryInput.reward
            else
                -- reward + discount(gamma) * max_a' Q(s',a')
                -- We are setting the Q-value for the action to  r + γmax a’ Q(s’, a’). The rest stay the same
                -- to give an error of 0 for those outputs.
                target[memoryInput.action] = memoryInput.reward + discount * nextStateMaxQ
            end
            -- Update the inputs and targets.
            inputs[i] = memoryInput.inputState
            targets[i] = target
        end
        return inputs, targets
    end

    return memory
end

return Memory
