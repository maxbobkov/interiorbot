import hashlib
import hmac
import json
from typing import Dict
from urllib.parse import parse_qsl


class InitDataError(ValueError):
    pass


def validate_init_data(init_data: str, bot_token: str) -> Dict[str, str]:
    if not init_data:
        raise InitDataError('init_data missing')

    data = dict(parse_qsl(init_data, keep_blank_values=True))
    received_hash = data.pop('hash', None)
    if not received_hash:
        raise InitDataError('hash missing')

    data_check_string = '\n'.join(f"{k}={data[k]}" for k in sorted(data))
    secret_key = hmac.new(b'WebAppData', bot_token.encode('utf-8'), hashlib.sha256).digest()
    calculated_hash = hmac.new(
        secret_key,
        data_check_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(calculated_hash, received_hash):
        raise InitDataError('invalid hash')

    return data


def get_user_id(init_data: str, bot_token: str) -> str:
    data = validate_init_data(init_data, bot_token)
    user_json = data.get('user')
    if not user_json:
        raise InitDataError('user missing')

    user = json.loads(user_json)
    user_id = user.get('id')
    if user_id is None:
        raise InitDataError('user id missing')

    return str(user_id)
