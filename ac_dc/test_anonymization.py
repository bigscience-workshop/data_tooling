import random
from anonymization import apply_regex_anonymization
from faker import Faker
from num2words import num2words

# We may need to include other test scenarios
# Wherever possible, test with faker


def main():
    test_suite = {"English": test_en, "Chinese": test_zh}
    for language, test_func in test_suite.items():
        print("Testing {}".format(language))
        test_func()
        print("==================================================")


def test_en():
    fake = Faker("en_US")
    sentences = [
        f"I am {num2words(random.randint(0,120))} years old, and she is {random.randint(0,120)} year-old",  # Age
        f"Sherry lives at {fake.street_address()}",  # Address
        f"My dad is a cancer fighter. Her grandma is suffering from alzheimer's",  # Disease
        f"Let me tell you a secret, Mr. Nguyen's SSN is {fake.ssn() if random.choice([True, False]) else fake.ssn().replace('-', '')}.",  # Government ID
        f"Dear Ian, the payment through {fake.credit_card_number()} has been successfully executed.",  # Credit card
    ]
    for sentence in sentences:
        print(
            apply_regex_anonymization(
                sentence=sentence, lang_id="en", anonymize_condition=True
            )
        )


def test_zh():
    fake = Faker("zh_CN")
    sentences = [
        f'我今年{num2words(random.randint(0,120), lang="ja")}歲, 而她去年{random.randint(0,120)}岁',  # Age
        f"我住在{fake.street_address()}",  # Address
        f"我爸是抗癌戰士。她奶奶有老人癡呆",  # Disease
        f"李雪妮小姐331125198402010129",  # Government ID
        f"先生，信用卡号{fake.credit_card_number()}已缴费成功",  # Credit card
    ]
    for sentence in sentences:
        print(
            apply_regex_anonymization(
                sentence=sentence, lang_id="zh", anonymize_condition=True
            )
        )


if __name__ == "__main__":
    main()
