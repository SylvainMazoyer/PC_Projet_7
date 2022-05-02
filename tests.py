import requests


URL = "http://127.0.0.1:5000/"


def test_hello(route="hello"):

    response = requests.get(URL + route)
    assert response.status_code == 200


def test_send_sku(route="send_sku"):

    response = requests.get(URL + route)
    assert response.status_code == 200
    sku_dict = response.json()
    assert isinstance(sku_dict, dict)

    # sku_list = [(i, j) for i, j in sku_dict.items()]
    # print(sku_list[:10])


def test_predict(route="predict", sku=100002):

    response = requests.get(URL + route, json={"sku": sku})
    if response.status_code != 200:
        print(response.text)
    print(response.json())


def test_shap(route="return_shap_data", sku=100002):

    response = requests.get(URL + route, json={"sku": sku})
    if response.status_code != 200:
        print(response.text)
    print(response.json())


if __name__ == "__main__":
    test_hello()
    test_send_sku()
    test_predict()
    test_shap()
