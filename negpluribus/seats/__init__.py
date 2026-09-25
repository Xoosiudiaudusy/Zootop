"""Seats for foreign bots: adapters that let an outside engine play at our table.

A seat is an ordinary ``Agent`` (``act(obs) -> Action``, ``end_hand``, ``reset``),
so every runner we have (``play_hand``, ``CashTable``, ``duplicate_match``)
accepts it unchanged.  Everything specific to the foreign bot (its input
format, its bet-size convention, its imports) stays inside the seat module.

* ``mozg.MozgAgent`` - the friend's engine from ``friends/mozg``.
"""
