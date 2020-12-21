{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import NeuralNetwork
import Text.CSV
import Data.Either
import Data.Tuple

main :: IO ()
main = do
    rawTrainData <- (fmap (map (map read) . fromRight []) . parseCSVFromFile) "res/mnist_test.csv" :: IO [[Double]]
    let processedData = map (fmap (\[a] -> take (round a) (repeat 0.0) ++ [1.0] ++ take (9 - round a) (repeat 0.0)) . swap . splitAt 1) rawTrainData
    let (trainData, testData) = splitAt 8000 processedData
    nn <- createStd [784, 16,16, 10] 0.5
    trainedNN <- train nn 30 (take 6000 trainData)

    print (accuracy trainedNN (take 1000 testData))
    let compare nn (a,b) = (eval trainedNN a, b)
    print (map (compare trainedNN) (take 10 testData))
