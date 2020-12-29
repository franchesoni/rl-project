import numpy as np

def test_CharacterTable():
    charTable = CharacterTable('abcd ', 10)
    input_C = 'da da'
    # test encode
    # first, 'd' = 5, 'a' = 1, ' ' = 0 and fill with zeros for 5 more characters
    expected_code1 = np.array([[0, 0, 0, 0, 1], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    expected_code2 = np.array([[0, 0, 0, 0, 1], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    actual_code1 = charTable.encode(input_C)
    actual_code2 = charTable.encode(input_C, maxlen=7)
    assert np.all(expected_code1 == actual_code1)
    assert np.all(expected_code2 == actual_code2)
    # test decode
    input_code1 = expected_code1
    actual_decoded_str = charTable.decode(input_code1)
    assert actual_decoded_str[:len(input_C)] == input_C  # ignore the padded spaces
    print('Great! CharacterTable works as expected')



def test_make_model():
    chars = '0123456789+ '
    input_C = '12   +   34'
    maxlen = 11
    max_digits = 5
    num_chars = len(sorted(set(chars)))
    charTable = CharacterTable(chars, maxlen)
    encoded_input = charTable.encode(input_C)

    hidden_size = 128

    input = Input(shape=(maxlen, num_chars))
    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
    # note: in a situation where your input sequences have a variable length,
    # use input_shape=(None, nb_feature).
    x = recurrent.LSTM(hidden_size)(input)  # outputs only one hidden_size vector
    # For the decoder's input, we repeat the encoded input for each time step
    x = RepeatVector(max_digits + 1)(x)  # at most one digit more than the maximum of digits of the sumands
    # The decoder RNN could be multiple layers stacked or a single layer
    x = recurrent.LSTM(hidden_size, return_sequences=True)(x)  # outputs every hidden state
    # For each of step of the output sequence, decide which character should be chosen
    x = TimeDistributed(Dense(num_chars, activation='softmax'))(x)

    model = Model(inputs=input, outputs=x)

    # test model
    output = model(np.array([encoded_input]))
    assert output.shape == (1, max_digits+1, num_chars)
    print('Great! The keras model outputs something that makes sense')



def test_softmax():
    input_ones = torch.ones((5, 4))
    assert (input_ones / 4 == torch.nn.functional.softmax(input_ones, dim=1)).byte().all()
    assert (input_ones / 5 == torch.nn.functional.softmax(input_ones, dim=0)).byte().all()
    print('Great! Softmax works as expected.')


def test_AdditionLSTM():
    # create input
    chars = '0123456789+ '
    input_C = '12+304'
    max_digits = 4
    maxlen = max_digits * 2 + 1
    charTable = CharacterTable(chars, maxlen)
    encoded_input = charTable.encode(input_C)

    # set as tensor, initialize and forward pass
    torch_input = torch.from_numpy(np.array([encoded_input])).float()
    model = AdditionLSTM(max_digits=max_digits)
    output = model(torch_input)
    assert output.shape == (max_digits+1, len(sorted(set(chars))))





if __name__ == '__main__':
    test_CharacterTable()
    # test_make_model()
    test_softmax()
    test_AdditionLSTM()

