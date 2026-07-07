#include <iostream>
#include <map>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>

#include "Matrix.h"
#include "Layer.h"
#include "Network.h"
#include "Activations.h"

struct train {
    std::map<std::string, int> indexedMap;
    std::map<std::string, int> authorMap;
    std::vector<std::pair<std::vector<std::string>, std::string>> rawMessages;;
    std::vector<std::pair<std::vector<float>, std::vector<float>>> vocabVector;

    std::string trainingData;
    std::string prevIndexedWords;
    std::string networkData;

    float trainingRate = 0.05f;
    Network network;



    train(std::string trainingData, std::string prevIndexedWords, std::string networkData) :
    prevIndexedWords(prevIndexedWords), trainingData(trainingData), networkData(networkData), network(trainingRate) {
        index_words();
        setup_training_data();
        to_float_data();


        network.add_layer(Layer(indexedMap.size(), 128, relu, relu_derivative));
        network.add_layer(Layer(128, 64, relu, relu_derivative));
        network.add_layer(Layer(64, authorMap.size(), sigmoid, sigmoid_derivative));


        if (networkData != "-")
            network.load(networkData, 1);

        train_on_data();
    }

    bool contains_number(const std::string& checkWord) {
        if (checkWord.empty()) return false;
        for (char c : checkWord) {
            if (std::isdigit(c))
                return true;
        }
        return false;
    }

    void index_words(){
        int index = 0, authorIndex = 0;
        std::string word, secondWord;
        if (prevIndexedWords != "-") {
            std::ifstream file(prevIndexedWords);
            file >> word;
            index = std::stoi(word);

            while (file>>word) {
                if (word == "a:") {
                    file>>word;
                    authorIndex = std::stoi(word);
                    break;
                }
                else {
                    file >> secondWord;
                    indexedMap[word] = std::stoi(secondWord);
                }
            }

            while (file>>word) {
                file >> secondWord;
                authorMap[word] = std::stoi(secondWord);
            }




            file.close();
        }

        if (trainingData != "-") {
            std::ifstream file(trainingData);
            while (file>>word) {
                if (word == "#eos") {
                    file>>word;
                    word = word.substr(1);
                    if (!word.empty() && word.back() == '-')
                        word.pop_back();

                    if (authorMap.find(word) == authorMap.end())
                        authorMap[word] = authorIndex++;
                }
                else {
                    for (char& c : word)
                        c = tolower(c);

                    if (!word.empty() && (word.front() == '"' || word.front() == '(')) {
                        std::string frontChar(1, word.front());
                        if (indexedMap.find(frontChar) == indexedMap.end())
                            indexedMap[frontChar] = index++;
                        word = word.substr(1);
                    }

                    if (word.back() == '.' || word.back() == ','
                    || word.back() == ':' || word.back() == ';' || word.back() == '"'
                    || word.back() == '?' || word.back() == '!'  || word.back() == ')') {
                        std::string punct(1, word.back());
                        word = word.substr(0, word.size() - 1);

                        if (contains_number(word))
                            word = "<NUMBER>";

                        if (!word.empty() && indexedMap.find(word) == indexedMap.end())
                            indexedMap[word] = index++;

                        if (indexedMap.find(punct) == indexedMap.end())
                            indexedMap[punct] = index++;
                    }else{
                        if (contains_number(word))
                            word = "<NUMBER>";
                        if (indexedMap.find(word) == indexedMap.end())
                            indexedMap[word] = index++;
                    }
                }
            }
            file.close();
        }
        if (prevIndexedWords == "-") {
            std::cout << "Give a name to these index word and author data: " << "\n";
            while (prevIndexedWords == "-" || prevIndexedWords.empty())
                std::cin >> prevIndexedWords;
        }
        std::ofstream indexedWordOutput(prevIndexedWords);
        indexedWordOutput << index << "\n";
        for (auto& pair : indexedMap)
            indexedWordOutput << pair.first << " " << pair.second << "\n";

        indexedWordOutput<<"a: "<<authorIndex<< "\n";
        for (auto& pair : authorMap)
            indexedWordOutput << pair.first << " " << pair.second << "\n";

        indexedWordOutput.close();
    }
    void setup_training_data() {
        std::ifstream file(trainingData);
        std::string word;
        std::vector<std::string> sentence;
        while (file>>word) {
            if (word == "#eos") {
                file>>word;
                word = word.substr(1);
                if (!word.empty() && word.back() == '-')
                    word.pop_back();
                rawMessages.push_back(std::pair<std::vector<std::string>, std::string>(sentence, word));
                sentence.clear();
            }
            else {
                for (char& c : word)
                    c = tolower(c);

                if (!word.empty() && (word.front() == '"' || word.front() == '(')) {
                    sentence.push_back(std::string(1, word.front()));
                    word = word.substr(1);
                }

                if (word.back() == '.' || word.back() == ','
                || word.back() == ':' || word.back() == ';' || word.back() == '"'
                || word.back() == '?' || word.back() == '!'  || word.back() == ')') {
                    std::string punct(1, word.back());
                    word = word.substr(0, word.size() - 1);

                    if (contains_number(word))
                        word = "<NUMBER>";

                    sentence.push_back(word);
                    sentence.push_back(punct);
                }else {
                    if (contains_number(word))
                        word = "<NUMBER>";
                    sentence.push_back(word);
                }
            }
        }
    }
    void to_float_data() {
        for (auto& [words, author] : rawMessages) {
            std::vector<float> data(indexedMap.size(), 0.0f);
            for (auto& word : words)
                if (indexedMap.count(word))
                    data[indexedMap.at(word)] = 1.0f;

            std::vector<float> authorData(authorMap.size(), 0.0f);
            if (authorMap.count(author))
                authorData[authorMap.at(author)] = 1.0f;

            vocabVector.push_back({data, authorData});
        }
    }
    void train_on_data() {
        int epochs = 0;
        std::cout << "How many training epoch (>10)?" << std::endl;
        while (epochs < 10) {
            std::cin >> epochs;
        }
        std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
        std::ofstream csvFile("../data/data.csv");
        csvFile<<"epoch,loss,clr\n";
        for (int epoch = 0; epoch < epochs; epoch++) {
            float totalLoss = 0.0f;

            std::shuffle(vocabVector.begin(), vocabVector.end(), rng);
            for (auto& [input, expected] : vocabVector) {
                Matrix inputMatrix(input, input.size(), 1);
                Matrix expectedMatrix(expected, expected.size(), 1);
                Matrix prediction = network.forward(inputMatrix);
                totalLoss += network.loss(prediction, expectedMatrix);
                network.backward(prediction, expectedMatrix, epoch);
            }
            if (epoch % 100 == 0) {
                std::cout << "+-----------------------------------------------+ "<<std::endl;
                std::cout << "Epoch " << epoch << " loss: " << totalLoss / vocabVector.size() <<std::endl;
                std::cout << "Current learning rate: " << trainingRate * (1.0f /(1.0f+0.0001f*epoch))  <<std::endl;
                std::cout << "+-----------------------------------------------+ "<<std::endl;
                csvFile<<epoch<<","<<totalLoss/vocabVector.size()<<","<<trainingRate * (1.0f /(1.0f+0.0001f*epoch))<<"\n";
            }

        }
        if (networkData == "-") {
            std::cout << "Give a name to the network: " << "\n";
            while (networkData == "-" || networkData.empty())
                std::cin >> networkData;
        }
        network.save(networkData);
    }
};
struct run {
    std::string model, vocab;
    std::map<std::string, int> indexedMap;
    std::map<std::string, int> authorMap;

    Network network;

    run(std::string model, std::string vocab) : model(model), vocab(vocab) {
        if (vocab == "-" || model == "-") {
            std::cout << "No vocabulary file or model has been inputed" << "\n";
            exit(0);
        }

        load_vocab();

        network.add_layer(Layer(indexedMap.size(), 128, relu, relu_derivative));
        network.add_layer(Layer(128, 64, relu, relu_derivative));
        network.add_layer(Layer(64, authorMap.size(), sigmoid, sigmoid_derivative));

        network.load(model, 0);

        enter_sentence();
    }

    bool contains_number(const std::string& checkWord) {
        if (checkWord.empty()) return false;
        for (char c : checkWord) {
            if (std::isdigit(c))
                return true;
        }
        return false;
    }
    void load_vocab() {
        std::ifstream file(vocab);
        std::string word, secondWord;
        file >> word;
        while (file>>word) {
            if (word == "a:") {
                file>>word;
                break;
            }else {
                file >> secondWord;
                indexedMap[word] = std::stoi(secondWord);
            }
        }

        while (file>>word) {
            file >> secondWord;
            authorMap[word] = std::stoi(secondWord);
        }

        file.close();
    }
    void enter_sentence() {
        std::string sentence = "";
        while (true) {
            std::cout << "Enter a sentence: " << "\n";
            std::getline(std::cin >> std::ws, sentence);

            if (sentence == "exit0") break;
            std::vector<float> sentenceData(indexedMap.size(), 0.0f);
            std::istringstream stream(sentence);
            std::string word;
            while (stream >> word) {
                for (char& c : word) c = tolower(c);

                if (!word.empty() && (word.front() == '"' || word.front() == '(')) {
                    std::string frontChar(1, word.front());
                    if (indexedMap.count(frontChar))
                        sentenceData[indexedMap.at(frontChar)] = 1.0f;
                    word = word.substr(1);
                }

                if (!word.empty() && (word.back() == '.' || word.back() == ',' ||
                word.back() == ':' || word.back() == ';' || word.back() == '"' ||
                word.back() == '?' || word.back() == '!'  || word.back() == ')')) {
                    std::string punct(1, word.back());
                    word = word.substr(0, word.size() - 1);

                    if (contains_number(word))
                        word = "<NUMBER>";

                    if (!word.empty() && indexedMap.count(word))
                        sentenceData[indexedMap.at(word)] = 1.0f;
                    if (indexedMap.count(punct))
                        sentenceData[indexedMap.at(punct)] = 1.0f;
                } else {
                    if (indexedMap.count(word))
                    sentenceData[indexedMap.at(word)] = 1.0f;
                }
            }

            Matrix input(sentenceData, sentenceData.size(), 1);
            Matrix result = network.forward(input);

            int author = 0;
            float maxLikelihood = 0.0f;
            for (int i = 0; i < (int)result.data.size(); i++) {
                if (result.data[i] > maxLikelihood) {
                    maxLikelihood = result.data[i];
                    author = i;
                }
            }

            std::cout << sentence << std::endl;
            for (auto& [name, index] : authorMap)
                if (index == author)
                    std::cout << "| Possibly by: "<<name << " | Likelihood: "<<maxLikelihood<<" |"<<std::endl;
        }
    }

};


int main() {
    std::cout << "Train or Run?\n";
    std::string input;
    std::cin >> input;

    std::string vocabFile, modelFile, trainingDataFile;

    if (input == "train") {
        while (trainingDataFile.empty()) {
            std::cout << "Training data file: \n";
            std::cin >> trainingDataFile;
        }
        if (vocabFile.empty()) {
            std::cout << "Previous vocabulary file? \"-\" if none \n";
            std::cin >> vocabFile;
        }
        if (modelFile.empty()) {
            std::cout << "model file? \"-\" if none \n";
            std::cin >> modelFile;
        }
        train(trainingDataFile, vocabFile, modelFile);
    }else if (input == "run") {
        while(vocabFile.empty() || vocabFile == "-") {
            std::cout << "Previous vocabulary file? required\"-\" if none \n";
            std::cin >> vocabFile;
        }
        while (modelFile.empty() || modelFile == "-") {
            std::cout << "model file? required \n";
            std::cin >> modelFile;
        }
        run(modelFile, vocabFile);
    }else {
        std::cerr << "Wrong input \n";
    }

    return 0;
}




