from pathlib import Path

BASE_DIR=Path("/documents")
#BASE_DIR.mkdir(parents=True, exist_ok=True)

documents = {
    "return_policy.md": """
# Return Policy

Customers can return most products within 10 days of delivery.

Electronics such as headphones, keyboards, smart watches, and mobile accessories
can be returned only within 7 days of delivery.

Products must be unused, undamaged, and returned with original packaging.

Items marked as final sale cannot be returned.

If the product is defective, the customer must upload images or videos as proof
before the return request is approved.
""",

    "refund_policy.md": """
# Refund Policy

Refunds are initiated after the returned product passes quality inspection.

For prepaid orders, the refund is usually processed within 5 to 7 business days.

For cash-on-delivery orders, customers must provide bank account details or UPI ID.

Shipping charges are non-refundable unless the product was damaged, defective,
or incorrectly delivered.

If the customer cancels before shipping, the full amount is refunded.
""",

    "shipping_policy.md": """
# Shipping Policy

Standard delivery usually takes 3 to 5 business days in metro cities.

Delivery to remote areas may take 7 to 10 business days.

Customers receive a tracking link once the order is shipped.

If delivery fails because the customer is unavailable, the courier partner will
make up to 2 additional delivery attempts.

Heavy appliances may require scheduled delivery.
""",

    "warranty_policy.md": """
# Warranty Policy

Warranty coverage depends on the product category and manufacturer.

Electronic accessories come with a 6-month limited warranty.

Home appliances usually come with a 1-year manufacturer warranty.

Warranty does not cover physical damage, water damage, misuse, or unauthorized repair.

Customers must provide the invoice to claim warranty support.
""",

    "cancellation_policy.md": """
# Cancellation Policy

Customers can cancel an order before it is shipped.

Once the order is shipped, cancellation is not allowed. The customer may place
a return request after delivery if the item is eligible.

For prepaid orders cancelled before shipping, the full refund is processed within
3 to 5 business days.

Orders containing personalized or made-to-order products cannot be cancelled
after production starts.
"""
}

for filename, content in documents.items():
    file_path = BASE_DIR/file_path
    file_path.write_text(content.strip(), encoding="utf-8")

