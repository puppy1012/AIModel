import numpy as np
selected_attributes = ['5_o_Clock_Shadow', 'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses', 
                       'Goatee', 'Gray_Hair', 'Male', 'Mustache', 'Pale_Skin', 'Smiling', 'Straight_Hair', 
                       'Wavy_Hair', 'Wearing_Hat', 'Young']
selected_attrs = [
    '5_o_Clock_Shadow', 'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 
    'Brown_Hair', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Male', 'Mustache', 
    'Pale_Skin', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat', 'Young'
]

interpolating_attributes = ['Young']

test_attributes = [
    ('Black_Hair', 1), ('Blond_Hair', 1), ('Brown_Hair', 1), 
    ('Male', 1), ('Young', 1), ('Young', -1)
]

inter_annos = np.zeros(
    (10 * len(interpolating_attributes), len(selected_attributes)), 
    dtype=np.float32
)

for i, attr in enumerate(interpolating_attributes):
    print("i: ", i, " attr: ", attr)
    index = selected_attributes.index(attr) # selected_attribues에서 attr이 몇 번째 인덱스인지 알려줌
    inter_annos[np.arange(10*i, 10*i+10), index] = np.linspace(0.1, 1, 10)