'''
Copyright (C) 2023 Fabio Bonassi

This file is part of ssnet.

ssnet is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ssnet is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU General Public License
along with ssnet.  If not, see <http://www.gnu.org/licenses/>.
'''

from pathlib import Path

import scipy.io as sio


class MatFileWriter:
    def __init__(self, path: str | Path) -> None:
        """
        Auxiliary class that orderly manages the saving of variables to a MATLAB file.

        Parameters
        ----------
        path : str | Path
            The path of the mat file.
        """
        self.path = Path(path).with_suffix('.mat')
        self._buffer = {}

    def push(self, **kwds):
        """
        Queue data to save.

        Raises
        ------
        ValueError
            When a key is already in the buffer
        """
        for key, item in kwds.items():
            if key in self._buffer:
                raise ValueError(f'Item {key} already exists in the buffer')
            self._buffer[key] = item
    
    def close(self):
        """
        Save the mat file.
        """
        sio.savemat(self.path, self._buffer)
