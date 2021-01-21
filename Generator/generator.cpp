#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>

#define ERR(source) (perror(source),                                 \
                     fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), \
                     exit(EXIT_FAILURE))

using namespace std;


//half of the generated vectors is created using bernoulli distribution
//the other half is created by changing bits so that every element has a pair
void generateRandomSequence(int vector_count, int vector_length, string fileName)
{

    ofstream myfile(fileName);
    vector<int> randomSequence;
    randomSequence.resize(vector_length);

    random_device rd;
    mt19937 generator(rd());
    bernoulli_distribution distribution(0.5);

    vector_count /= 2;

    if (myfile.is_open())
    {
        while (vector_count--)
        {
            generate(randomSequence.begin(), randomSequence.end(), [&generator, &distribution] { return distribution(generator); });

            for (auto z : randomSequence)
            {
                myfile << z;
            }

            myfile << endl;
            auto c = vector_count % vector_length;

            randomSequence[c] = 1 - randomSequence[c];
            
            for (auto z : randomSequence)
            {
                myfile << z;
            }
            myfile << endl;
        }
    }
    else
    {
        ERR("Could not open the file");
    }
}

//the function accepts two arguments N,L (number, length) of vectors to generate
int main(int argc, char **argv)
{

    if (argc != 3)
    {
        ERR("You need to provide number and length of vectors to be generated");
    }

    int vector_count = atoi(argv[1]);
    int vector_length = atoi(argv[2]);

    generateRandomSequence(vector_count, vector_length, "input.txt");

    return 0;
}