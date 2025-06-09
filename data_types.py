"""
This file contains the type definitions for the Roll Call FactBase API.

Despite the API route endpoint name, it also includes posts from Truth Social.

Request Example:

```http
GET https://api.factsquared.com/json/factba.se-trump-social.php?q=&page=1&sort=date&sort_order=asc
```

Response Example:

```json
{
    "meta": {
        "milliseconds": 27,
        "total_hits": 86520,
        "page_size": 50,
        "page": 1,
        "from": 1,
        "page_count": 1731,
        "pagination": {
            "previous_page": null,
            "next_page": "http://api.factsquared.com/json/factba.se-trump-social.php?q=&page=2&sort=date&sort_order=asc"
        }
    },
    "data": [
        {
            "browse_flag": true,
            "date": "2009-05-04T14:54:25-04:00",
            "document_id": "1698308935",
            "image_url": "https://media-cdn.factba.se/realdonaldtrump-twitter/1698308935.jpg",
            "media_type": "Text",
            "sequence": 1,
            "speaker": "Donald Trump",
            "speaker_id": "Q22686",
            "subject": "Donald Trump",
            "text": "Be sure to tune in and watch Donald Trump on Late Night with David Letterman as he presents the Top Ten List tonight!",
            "type": "Social",
            "word_count": 23,
            "deleted_flag": false,
            "client": "Twitter Web Client",
            "handle": "realDonaldTrump",
            "id": "1698308935",
            "platform": "Twitter",
            "post_url": "https://x.com/realdonaldtrump/status/1698308935",
            "social": {
                "author": "realDonaldTrump",
                "deleted_date": null,
                "favorite_count": 936,
                "hashtags": [],
                "in_reply_to_screen_name": null,
                "media_count": null,
                "media_durations": null,
                "media_filenames": [],
                "media_urls": [],
                "post_text": "Be sure to tune in and watch Donald Trump on Late Night with David Letterman as he presents the Top Ten List tonight!",
                "quote_flag": false,
                "repost_count": 512,
                "repost_flag": false,
                "urls": [],
                "user_mentions": [],
                "repost_id": null,
                "quote_id": null
            },
            "search_id": "1698308935",
            "score": null
        },
        {
            "browse_flag": true,
            "date": "2025-06-04T14:00:37-04:00",
            "document_id": "114626398631443870",
            "image_url": "https://media-cdn.factba.se/realdonaldtrump-truthsocial/114626398631443870.jpg",
            "media_type": "Text",
            "sequence": 1,
            "speaker": "Donald Trump",
            "speaker_id": "Q22686",
            "subject": "Donald Trump",
            "text": "https:// thehill.com/business/5320379-u s-housing-finance-chief-tells-powell-to-lower-interest-rates/",
            "type": "Social",
            "word_count": 3,
            "deleted_flag": false,
            "account_url": "https://truthsocial.com/@realDonaldTrump",
            "handle": "realDonaldTrump",
            "id": "114626398631443870",
            "platform": "Truth Social",
            "post_url": "https://truthsocial.com/@realDonaldTrump/posts/114626398631443870",
            "social": {
                "author": "realDonaldTrump",
                "deleted_date": null,
                "favorite_count": 45,
                "hashtags": [],
                "in_reply_to_screen_name": null,
                "media_count": null,
                "media_text": [],
                "media_durations": [],
                "media_filenames": [],
                "media_filesizes": [],
                "media_urls": [],
                "post_html": "\u003cp\u003e\u003ca href=\"https://thehill.com/business/5320379-us-housing-finance-chief-tells-powell-to-lower-interest-rates/\" rel=\"nofollow noopener noreferrer\" target=\"_blank\"\u003e\u003cspan class=\"invisible\"\u003ehttps://\u003c/span\u003e\u003cspan class=\"ellipsis\"\u003ethehill.com/business/5320379-u\u003c/span\u003e\u003cspan class=\"invisible\"\u003es-housing-finance-chief-tells-powell-to-lower-interest-rates/\u003c/span\u003e\u003c/a\u003e\u003c/p\u003e",
                "post_text": "https:// thehill.com/business/5320379-u s-housing-finance-chief-tells-powell-to-lower-interest-rates/",
                "quote_flag": false,
                "quote_id": null,
                "repost_count": 14,
                "repost_flag": false,
                "repost_id": null,
                "urls": [
                    "https://thehill.com/business/5320379-us-housing-finance-chief-tells-powell-to-lower-interest-rates/"
                ],
                "user_mentions": [],
                "visibility": "public"
            },
            "search_id": "114626398631443870",
            "score": null
        },
    ],
    "stats": {
        "deleted": { "deleted": 2544, "not_deleted": 83976 },
        "platform": { "truth_social": 27713, "x_twitter": 58807, "unknown": 0 }
    }
}
```
"""

from enum import Enum
from typing import List, Any, Literal, Optional
from pydantic import BaseModel, Field


################################################################################
# Request


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


class RequestParams(BaseModel):
    page: int
    sort: Literal["date"]
    sort_order: SortOrder


################################################################################
# Response


class MetaPagination(BaseModel):
    previous_page: Optional[str]
    next_page: Optional[str]


class Meta(BaseModel):
    milliseconds: int
    total_hits: int
    page_size: int
    page: int
    from_: int = Field(..., alias="from")
    page_count: int
    pagination: MetaPagination


class DataItemCommonProps(BaseModel):
    browse_flag: bool
    date: str
    "Example: '2025-06-04T14:00:37-04:00' (ET Timezone)."
    document_id: str
    image_url: str
    media_type: str
    sequence: int
    speaker: str
    speaker_id: str
    subject: str
    text: str
    type: str
    word_count: int
    deleted_flag: bool
    handle: str
    id: str
    post_url: str
    search_id: str
    score: Any


class SocialForTruthSocial(BaseModel):
    author: str
    deleted_date: Optional[str]
    favorite_count: int
    hashtags: List[str]
    in_reply_to_screen_name: Any
    media_count: Any
    media_durations: Any
    media_filenames: List[str]
    media_filesizes: List[str]
    media_urls: List[str]
    post_text: str
    quote_flag: bool
    repost_count: int
    repost_flag: bool
    urls: List[str]
    user_mentions: List[str]
    repost_id: Optional[str]
    quote_id: Optional[str]


class SocialForTwitter(BaseModel):
    author: str
    deleted_date: Optional[str]
    favorite_count: Optional[int]
    hashtags: List[str]
    in_reply_to_screen_name: Any
    media_count: Any
    media_durations: Any
    media_filenames: List[str]
    media_urls: List[str]
    post_text: str
    quote_flag: bool
    repost_count: Optional[int]
    repost_flag: bool
    urls: List[str]
    user_mentions: List[str]
    repost_id: Optional[str]
    quote_id: Optional[str]


class DataItemForTruthSocial(DataItemCommonProps):
    account_url: str
    platform: Literal["Truth Social"]
    social: SocialForTruthSocial


class DataItemForTwitter(DataItemCommonProps):
    client: Optional[str] = None
    platform: Literal["Twitter"]
    social: SocialForTwitter


DataItem = DataItemForTruthSocial | DataItemForTwitter


class StatsDeleted(BaseModel):
    deleted: int
    not_deleted: int


class StatsPlatform(BaseModel):
    truth_social: int
    x_twitter: int
    unknown: int


class Stats(BaseModel):
    deleted: StatsDeleted
    platform: StatsPlatform


class ResponseBody(BaseModel):
    meta: Meta
    data: list[DataItem]
    stats: Stats
