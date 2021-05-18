import pytest

from typing import List

from starlette.testclient import TestClient
from starlette.status import HTTP_200_OK
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

test_input1 = ['first data','second data','third data']
test_input2 = ['tweet','tweet2','tweet3','tweet4','tweet5']
@pytest.mark.parametrize("input", [test_input1, test_input2])
def test_predict(input: List[str], test_client: TestClient):
    response = test_client.post("/classify", json={'data': input})
    assert response.status_code == HTTP_200_OK
    assert len(response.json()['data']) == len(input)

@pytest.mark.parametrize("input", [[]])
def test_predict_zero_length_input(input: List[str], test_client: TestClient):
    response = test_client.post("/classify", json={'data': input})
    assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

over_limit = ['test_data' for _ in range(0,120)]
@pytest.mark.parametrize("input", [over_limit])
def test_predict_120_length_input(input: List[str], test_client: TestClient):
    response = test_client.post("/classify", json={'data': input})
    assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY