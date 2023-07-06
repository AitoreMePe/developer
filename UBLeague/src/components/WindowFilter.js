import React, { useState } from 'react';

const WindowFilter = ({ data, filterKeys }) => {
  const [inputData, setInputData] = useState({});

  const handleInputChange = (e) => {
    setInputData({
      ...inputData,
      [e.target.name]: e.target.value,
    });
  };

  const filteredData = data.filter((item) => {
    for (let key in inputData) {
      if (item[key] === undefined || !item[key].includes(inputData[key])) {
        return false;
      }
    }
    return true;
  });

  return (
    <div>
      {filterKeys.map((key) => (
        <div key={key}>
          <label>{key}</label>
          <input
            type="text"
            name={key}
            onChange={handleInputChange}
          />
        </div>
      ))}
      <div>
        {filteredData.map((item, index) => (
          <div key={index}>
            {filterKeys.map((key) => (
              <p key={key}>{item[key]}</p>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};

export default WindowFilter;